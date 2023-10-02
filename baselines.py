import torch, numpy as np, copy
import warnings
from utils import LinearSchedule, minigridobs2tensor, get_cpprb, RL_AGENT


class RW_AGENT(RL_AGENT):
    def __init__(self, env, gamma=0.99, seed=42, **kwargs):
        super(RW_AGENT, self).__init__(env, gamma, seed)
        self.steps_interact = 0  # steps_interact denotes the number of agent-env interactions
        self.time_learning_starts = 20000

    def decide(self, *args, **kwargs):
        """
        input observation and output action
        some through the computations of the policy network
        """
        return self.action_space.sample()

    def step(self, *args, **kwargs):
        self.steps_interact += 1


class DQN_BASE(RL_AGENT):
    def __init__(
        self,
        env,
        network_policy,
        gamma=0.99,
        clip_reward=True,
        exploration_fraction=0.02,
        epsilon_final_train=0.01,
        epsilon_eval=0.0,
        steps_total=50000000,
        size_buffer=1000000,
        prioritized_replay=True,
        func_obs2tensor=minigridobs2tensor,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        seed=42,
    ):
        super(DQN_BASE, self).__init__(env, gamma, seed)

        self.clip_reward = clip_reward
        self.schedule_epsilon = LinearSchedule(
            schedule_timesteps=int(exploration_fraction * steps_total),
            initial_p=1.0,
            final_p=epsilon_final_train,
        )
        self.epsilon_eval = epsilon_eval
        self.device = device

        self.network_policy = network_policy.to(self.device)

        self.steps_interact, self.steps_total = (
            0,
            steps_total,
        )  # steps_interact denotes the number of agent-env interactions

        self.step_last_print, self.time_last_print = 0, None

        self.obs2tensor = func_obs2tensor

        self.prioritized_replay = prioritized_replay
        self.rb = get_cpprb(env, size_buffer, prioritized=self.prioritized_replay)
        if self.prioritized_replay:
            self.size_batch_rb = 64
            self.batch_rb = get_cpprb(env, self.size_batch_rb, prioritized=False)
        if self.prioritized_replay:
            self.schedule_beta_sample_priorities = LinearSchedule(steps_total, initial_p=0.4, final_p=1.0)

    def add_to_buffer(self, batch):
        if self.prioritized_replay:
            self.batch_rb.add(**batch)
            if self.batch_rb.get_stored_size() >= self.size_batch_rb:  # NOTE(H): calculate priorities in batches
                batch = self.batch_rb.get_all_transitions()
                self.batch_rb.clear()
                (
                    batch_obs_curr,
                    batch_action,
                    batch_reward,
                    batch_obs_next,
                    batch_done,
                    weights,
                    batch_idxes,
                ) = self.process_batch(batch, prioritized=False)
                priorities = self.calculate_priorities(
                    batch_obs_curr,
                    batch_action,
                    batch_reward,
                    batch_obs_next,
                    batch_done,
                    error_absTD=None,
                )
                self.rb.add(**batch, priorities=priorities)
        else:
            self.rb.add(**batch)

    def calculate_TD_error(self, batch_obs_curr, batch_action, batch_reward, batch_obs_next, batch_done, type="abs"):
        with torch.no_grad():
            predicted_Q_next = self.network_policy(batch_obs_next, scalarize=True)
            action_next = torch.argmax(predicted_Q_next.detach(), dim=1, keepdim=True)
            predicted_target_Q = self.network_target(batch_obs_next, scalarize=True)
            values_next = predicted_target_Q.gather(1, action_next)
            values_next = torch.where(
                batch_done,
                torch.tensor(0.0, dtype=torch.float32, device=self.device),
                values_next,
            )
            target_TD = (batch_reward + self.gamma * values_next).detach()
        if type == "l1":
            values_curr = self.network_policy(batch_obs_curr, scalarize=True).gather(1, batch_action)
            return torch.nn.functional.l1_loss(values_curr, target_TD, reduction="none")
        elif type == "kld":
            value_logits_curr = self.network_policy(batch_obs_curr, scalarize=False)[torch.arange(batch_obs_curr.shape[0]), batch_action.squeeze()]
            with torch.no_grad():
                value_dist_target = self.network_policy.estimator_Q.histogram_converter.to_histogram(target_TD)
            return torch.nn.functional.kl_div(torch.log_softmax(value_logits_curr, -1), value_dist_target.detach(), reduction="none").sum(-1, keepdims=True)
        elif type == "huber":
            values_curr = self.network_policy(batch_obs_curr, scalarize=True).gather(1, batch_action)
            return torch.nn.functional.smooth_l1_loss(values_curr, target_TD, reduction="none")
        else:
            raise NotImplementedError("what is this loss type?")

    @torch.no_grad()
    def calculate_priorities(self, batch_obs_curr, batch_action, batch_reward, batch_obs_next, batch_done, error_absTD=None):
        if error_absTD is None:
            error_absTD = self.calculate_TD_error(batch_obs_curr, batch_action, batch_reward, batch_obs_next, batch_done, type="l1")
        else:
            assert error_absTD.shape[0] == batch_reward.shape[0]
        new_priorities = error_absTD.detach().cpu().numpy() + 1e-6
        return new_priorities

    @torch.no_grad()
    def process_batch(self, batch, prioritized=False):
        # even with prioritized replay, one would still want to process a batch without the priorities
        if prioritized:
            batch_obs_curr, batch_action, batch_reward, batch_done, batch_obs_next, weights, batch_idxes = batch.values()
            weights = torch.tensor(weights, dtype=torch.float32, device=self.device).reshape(-1, 1)
        else:
            batch_obs_curr, batch_action, batch_reward, batch_done, batch_obs_next = batch.values()
            weights, batch_idxes = None, None

        batch_reward = torch.tensor(batch_reward, dtype=torch.float32, device=self.device).reshape(-1, 1)
        batch_done = torch.tensor(batch_done, dtype=torch.bool, device=self.device).reshape(-1, 1)
        batch_action = torch.tensor(batch_action, dtype=torch.int64, device=self.device).reshape(-1, 1)

        batch_obs_curr, batch_obs_next = self.obs2tensor(batch_obs_curr, device=self.device), self.obs2tensor(batch_obs_next, device=self.device)
        if self.clip_reward:  # this is a DQN-specific thing
            batch_reward = torch.sign(batch_reward)
        return batch_obs_curr, batch_action, batch_reward, batch_obs_next, batch_done, weights, batch_idxes

    def decide(self, obs, eval=False, env=None, writer=None, random_walk=False):
        """
        input observation and output action
        some through the computations of the policy network
        """
        if np.random.random() > float(eval) * self.epsilon_eval + (1 - float(eval)) * self.schedule_epsilon.value(self.steps_interact):
            with torch.no_grad():
                return int(torch.argmax(self.network_policy(self.obs2tensor(obs, device=self.device))))
        else:  # explore
            return self.action_space.sample()

    def step(self, obs_curr, action, reward, obs_next, done, eval=False, writer=None):
        if obs_next is not None:
            sample = {"obs": np.array(obs_curr), "act": action, "rew": reward, "done": done, "next_obs": np.array(obs_next)}
            self.add_to_buffer(sample)
        self.steps_interact += 1


class DQN(DQN_BASE):
    def __init__(
        self,
        env,
        network_policy,
        gamma=0.99,
        clip_reward=True,
        exploration_fraction=0.02,
        epsilon_final_train=0.01,
        epsilon_eval=0.0,
        steps_total=50000000,
        size_buffer=1000000,
        prioritized_replay=True,
        type_optimizer="Adam",
        lr=5e-4,
        eps=1.5e-4,
        time_learning_starts=20000,
        freq_targetsync=8000,
        freq_train=4,
        size_batch=32,
        func_obs2tensor=minigridobs2tensor,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        seed=42,
    ):
        super(DQN, self).__init__(
            env,
            network_policy,
            gamma=gamma,
            clip_reward=clip_reward,
            exploration_fraction=exploration_fraction,
            epsilon_final_train=epsilon_final_train,
            epsilon_eval=epsilon_eval,
            steps_total=steps_total,
            size_buffer=size_buffer,
            prioritized_replay=prioritized_replay,
            func_obs2tensor=func_obs2tensor,
            device=device,
            seed=seed,
        )

        self.optimizer = eval("torch.optim.%s" % type_optimizer)(self.network_policy.parameters(), lr=lr, eps=eps)

        # initialize target network
        self.network_target = copy.deepcopy(self.network_policy)
        self.network_target.to(self.device)
        for param in self.network_target.parameters():
            param.requires_grad = False
        self.network_target.eval()
        for module in self.network_target.modules():
            module.eval()

        self.size_batch = size_batch
        self.time_learning_starts = time_learning_starts
        self.freq_train = freq_train
        self.freq_targetsync = freq_targetsync
        self.step_last_update = self.time_learning_starts - self.freq_train
        self.step_last_targetsync = self.time_learning_starts - self.freq_targetsync

    def step(self, obs_curr, action, reward, obs_next, done, eval=False, writer=None):
        """
        an agent step: in this step the agent does whatever it needs
        """
        if obs_next is not None:
            sample = {
                "obs": np.array(obs_curr),
                "act": action,
                "rew": reward,
                "done": done,
                "next_obs": np.array(obs_next),
            }
            self.add_to_buffer(sample)
        if self.steps_interact >= self.time_learning_starts:
            if self.rb.get_stored_size() >= self.size_batch and (self.steps_interact - self.step_last_update) >= self.freq_train:
                self.update(writer=writer)
                self.step_last_update += self.freq_train
            if (self.steps_interact - self.step_last_targetsync) >= self.freq_targetsync:
                self.sync_parameters()
                self.step_last_targetsync += self.freq_targetsync
        self.steps_interact += 1

    # @profile
    def update(self, batch=None, writer=None):
        """
        update the parameters of the DQN model using the weighted sampled Bellman error
        """
        if batch is None:
            if self.prioritized_replay:
                batch = self.rb.sample(
                    self.size_batch,
                    beta=self.schedule_beta_sample_priorities.value(self.steps_interact),
                )
            else:
                batch = self.rb.sample(self.size_batch)
        batch_obs_curr, batch_action, batch_reward, batch_obs_next, batch_done, weights, batch_idxes = self.process_batch(
            batch, prioritized=self.prioritized_replay
        )

        type_TD_loss = "huber"
        dict_head = self.network_policy.estimator_Q.dict_head
        if dict_head["name"] == "Q" and dict_head["dist_out"]:
            type_TD_loss = "kld"

        error_TD = self.calculate_TD_error(batch_obs_curr, batch_action, batch_reward, batch_obs_next, batch_done, type=type_TD_loss)

        if self.prioritized_replay:
            assert weights is not None
            error_TD_weighted = (error_TD * weights).mean()  # kaixhin's rainbow implementation used mean()
        else:
            error_TD_weighted = error_TD.mean()

        self.optimizer.zero_grad()
        error_TD_weighted.backward()

        # gradient clipping
        torch.nn.utils.clip_grad_value_(self.network_policy.parameters(), 1.0)
        self.optimizer.step()

        # update prioritized replay, if used
        if self.prioritized_replay:
            new_priorities = self.calculate_priorities(batch_obs_curr, batch_action, batch_reward, batch_obs_next, batch_done, error_absTD=None)
            self.rb.update_priorities(batch_idxes, new_priorities.squeeze())

        if writer is not None:
            writer.add_scalar("Train/error_TD", error_TD_weighted.detach().cpu().numpy(), self.step_last_update)

    def sync_parameters(self):
        """
        synchronize the parameters of self.network_policy and self.network_target
        this is hard sync, maybe a softer version is going to do better
        """
        self.network_target.load_state_dict(self.network_policy.state_dict())
        for param in self.network_target.parameters():
            param.requires_grad = False
        self.network_target.eval()
        print("policy-target parameters synced")


class DQN_NETWORK(torch.nn.Module):
    def __init__(self, encoder, estimator_Q, binder=None):
        super(DQN_NETWORK, self).__init__()
        self.encoder, self.estimator_Q = encoder, estimator_Q
        self.binder = binder

    def forward(self, obs, scalarize=True):
        state = self.encoder(obs)
        if self.binder is None:
            state_local = state
        else:
            state_local = self.binder(state, state)
        return self.estimator_Q(state_local, scalarize=scalarize)

    def parameters(self):
        parameters = []
        parameters += list(self.encoder.parameters())
        if self.binder is not None:
            parameters += list(self.binder.parameters())
        parameters += list(self.estimator_Q.parameters())
        return parameters


def create_RW_agent(args, env, **kwargs):
    return RW_AGENT(env, args.gamma, args.seed)


def create_DQN_agent(args, env, dim_embed, num_actions, device=None):
    if device is None:
        if torch.cuda.is_available() and not args.force_cpu:
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
            warnings.warn("agent created on cpu")
    from models import Encoder_MiniGrid, Binder_MiniGrid, Predictor_MiniGrid

    if args.activation == "relu":
        activation = torch.nn.ReLU
    elif args.activation == "elu":
        activation = torch.nn.ELU
    elif args.activation == "leakyrelu":
        activation = torch.nn.LeakyReLU
    elif args.activation == "silu":
        activation = torch.nn.SiLU

    encoder = Encoder_MiniGrid(dim_embed, sample_obs=env.reset(), norm=bool(args.layernorm), append_pos=bool(args.append_pos), activation=activation)
    encoder.to(device)

    sample_input = encoder(minigridobs2tensor(env.obs_curr))

    binder = Binder_MiniGrid(
        sample_input,
        len_rep=args.len_rep,
        norm=bool(args.layernorm),
        activation=activation,
        num_heads=args.num_heads,
        size_bottleneck=args.size_bottleneck,
        type_arch=args.arch_enc,
    )
    binder.to(device)

    dict_head_Q = {
        "name": "Q",
        "len_predict": num_actions,
        "dist_out": True,
        "value_min": args.value_min,
        "value_max": args.value_max,
        "atoms": args.atoms_value,
        "classify": False,
    }
    estimator_Q = Predictor_MiniGrid(
        num_actions,
        len_input=binder.len_out,
        depth=args.depth_hidden,
        width=args.width_hidden,
        norm=bool(args.layernorm),
        activation=activation,
        dict_head=dict_head_Q,
    )
    estimator_Q.to(device)

    agent = DQN(
        env,
        DQN_NETWORK(encoder, estimator_Q, binder=binder),
        gamma=args.gamma,
        steps_total=args.steps_max,
        prioritized_replay=bool(args.prioritized_replay),
        lr=args.lr,
        size_batch=args.size_batch,
        device=device,
        seed=args.seed,
    )
    return agent
