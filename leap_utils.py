import os, sys

sys.path.insert(1, os.getcwd())
import torch, numpy as np
from modules import ResidualBlock
from utils import dijkstra
from models import CVAE_MiniGrid_Separate2, Embedder_MiniGrid_BOW
from modules import TopKMultiheadAttention


class Binder_LEAP(torch.nn.Module):
    def __init__(self, cvae, len_code, len_rep, size_input, activation=torch.nn.ReLU):
        super(Binder_LEAP, self).__init__()
        self.len_code = len_code
        self.size_input = size_input
        self.cvae = cvae
        self.len_rep = len_rep
        self.len_out = 2 * len_rep
        self.activation = activation

        self.embedder = Embedder_MiniGrid_BOW(dim_embed=16, width=size_input, height=size_input, channels_obs=2, ebd_pos=False)
        self.fuser = torch.nn.Sequential(
            ResidualBlock(len_in=16, width=None, kernel_size=3, depth=2, stride=1, padding=1, activation=activation),
            torch.nn.Conv2d(16, len_rep, kernel_size=8, stride=1, padding=0),
        )
        self.register_buffer("query", torch.zeros(1, 1, len_rep))
        self.attn = TopKMultiheadAttention(
            embed_dim=len_rep,
            num_heads=1,
            kdim=len_rep,
            vdim=len_rep,
            batch_first=True,
            dropout=0.0,
            size_bottleneck=4,
            no_out_proj=True,
        )
        self.layer_norm_1 = torch.nn.LayerNorm(len_rep)
        self.layer_norm_2 = torch.nn.LayerNorm(len_rep)

    def to(self, device):
        super().to(device)
        self.embedder.to(device)
        self.fuser.to(device)
        self.query = self.query.to(device)
        self.attn.to(device)
        self.layer_norm_1.to(device)
        self.layer_norm_2.to(device)

    def parameters(self):
        parameters = []
        parameters += list(self.embedder.parameters())
        parameters += list(self.fuser.parameters())
        parameters += list(self.attn.parameters())
        parameters += list(self.layer_norm_1.parameters())
        parameters += list(self.layer_norm_2.parameters())
        return parameters

    def extract_local_field(self, fields):
        size_batch = fields.shape[0]
        fields = fields.permute(0, 2, 3, 1).reshape(size_batch, -1, self.len_rep)
        fields = self.layer_norm_1(fields)
        state_local, _ = self.attn(self.query.expand(size_batch, 1, self.len_rep), fields, fields, need_weights=False)
        state_local = self.layer_norm_2(state_local)
        state_local = self.activation()(state_local)
        state_local = state_local.reshape(size_batch, self.len_rep)
        return state_local

    def forward_train(self, obses_pred_curr, obses_pred_targ, return_curr=False):
        size_batch = obses_pred_curr.shape[0]
        obses_curr_targ = torch.cat([obses_pred_curr, obses_pred_targ], 0)
        fields_curr_targ = self.fuser(self.embedder(obses_curr_targ))
        states_local_curr_targ = self.extract_local_field(fields_curr_targ)
        state_local_curr, state_local_targ = torch.split(states_local_curr_targ, [size_batch, size_batch], dim=0)
        state_binded = torch.cat([state_local_curr, state_local_targ], dim=-1)
        if return_curr:
            return state_binded, state_local_curr
        else:
            return state_binded

    def forward_single(self, obses):
        fields_curr_targ = self.fuser(self.embedder(obses))
        return self.extract_local_field(fields_curr_targ)


import torch, numpy as np, copy
import warnings
from utils import LinearSchedule, minigridobs2tensor, RL_AGENT, process_batch


class LEAP_NETWORK(torch.nn.Module):
    # def __init__(self, binder, estimator_Q, estimator_distance, estimator_omega, cvae):
    def __init__(self, binder, estimator_distance, estimator_omega, cvae):
        super(LEAP_NETWORK, self).__init__()
        self.binder = binder
        # self.estimator_Q = estimator_Q
        self.estimator_distance = estimator_distance
        self.cvae = cvae
        self.estimator_omega = estimator_omega

    def to(self, device):
        super().to(device)
        self.binder.to(device)
        # self.estimator_Q.to(device)
        self.estimator_distance.to(device)
        self.cvae.to(device)
        self.estimator_omega.to(device)

    def parameters(self):
        parameters = []
        parameters += list(self.binder.parameters())
        # parameters += list(self.estimator_Q.parameters())
        parameters += list(self.estimator_distance.parameters())
        parameters += list(self.estimator_omega.parameters())
        return parameters


class LEAP_BASE(RL_AGENT):
    def __init__(
        self,
        env,
        network_policy,
        vae_discrete=True,
        freq_plan=16,
        num_waypoints=5,
        steps_per_subgoal=8,
        dist_cutoff=8,
        gamma=0.99,
        clip_reward=True,
        exploration_fraction=0.02,
        epsilon_final_train=0.01,
        epsilon_eval=0.001,
        steps_total=50000000,
        prioritized_replay=True,
        func_obs2tensor=minigridobs2tensor,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        seed=42,
        gamma_int=0.95,
        hrb=None,
        silent=False,
        suppress_delusion=False,
    ):
        super(LEAP_BASE, self).__init__(env, gamma, seed)
        self.vae_discrete = bool(vae_discrete)
        self.clip_reward = clip_reward
        self.schedule_epsilon = LinearSchedule(
            schedule_timesteps=int(exploration_fraction * steps_total),
            initial_p=1.0,
            final_p=epsilon_final_train,
        )
        self.epsilon_eval = epsilon_eval

        self.device = device

        self.freq_plan, self.step_last_planned = freq_plan, 0
        self.num_waypoints = num_waypoints
        self.steps_per_subgoal = steps_per_subgoal

        self.network_policy = network_policy
        self.network_target = self.network_policy

        self.cvae = self.network_policy.cvae

        self.gamma_int = gamma_int

        # if self.optimal_policy:
        #     assert self.cvae is None or self.optimal_plan, "no optimal policy for non-existing states"

        self.encoder_wp = lambda obs: self.cvae.encode_from_obs(obs).reshape(obs.shape[0], -1).squeeze_().cpu().numpy()
        self.decoder_wp = lambda code, obs: self.cvae.decode_to_obs(code, obs)

        self.dist_cutoff = dist_cutoff

        self.steps_interact, self.steps_total = 0, steps_total  # steps_interact denotes the number of agent-env interactions
        self.steps_processed = 0

        self.step_last_print, self.time_last_print = 0, None

        self.obs2tensor = lambda obs: func_obs2tensor(obs, device=self.device)

        self.prioritized_replay = prioritized_replay
        self.hrb = hrb
        if self.prioritized_replay:
            self.schedule_beta_sample_priorities = LinearSchedule(steps_total, initial_p=0.4, final_p=1.0)
        self.silent = silent

        self.waypoints_existing, self.wp_graph_curr = None, None
        self.suppress_delusion = bool(suppress_delusion)

        self.on_episode_end(eval=True)  # NOTE: do not call hrb.on_episode_end() here when there is no experience

    def add_to_buffer(self, batch):
        self.hrb.add(**batch)

    @torch.no_grad()
    def process_batch(self, batch, prioritized=False, with_targ=False):
        return process_batch(
            batch, prioritized=prioritized, with_targ=with_targ, device=self.device, obs2tensor=minigridobs2tensor, clip_reward=self.clip_reward, aux=False
        )

    @torch.no_grad()
    def reinit_plan(self):
        self.waypoint_last_reached = None
        self.steps_this_subgoal = 0
        self.waypoint_targ = None
        self.state_wp_targ = None
        self.replan = True

    @torch.no_grad()
    def on_episode_end(self, eval=False):
        self.reinit_plan()
        self.steps_episode = 0

        self.code_goal = None
        self.episode_for_debug = np.random.rand() < 0.01 or eval
        self.num_subgoals_finished = 0
        self.obses_intermediate_subgoals = None
        self.obs_goal_tensor = None
        self.obs_targ = None

        self.replan = True
        if self.wp_graph_curr is not None:
            del self.wp_graph_curr
            self.wp_graph_curr = None
        self.num_planning_triggered = 0
        self.num_planning_triggered_timeout = 0
        self.num_waypoints_reached = 0
        if self.hrb is not None and not eval:
            self.hrb.on_episode_end()

    # @profile
    def calculate_loss(
        self,
        batch_obs_curr,
        batch_action,
        batch_reward,
        batch_obs_next,
        batch_done,
        batch_obs_targ,
        calculate_priorities=True,
        type_priorities="kl",
        debug=False,
        writer=None,
    ):
        debug = debug and writer is not None
        size_batch = batch_obs_curr.shape[0]
        with torch.no_grad():
            batch_targ_reached = (batch_obs_next == batch_obs_targ).reshape(size_batch, -1).all(-1)
            mask_dead = torch.logical_and(batch_done, ~batch_targ_reached)
            batch_obs_next_targ = torch.cat([batch_obs_next, batch_obs_targ], 0)
            batch_obs_curr_next_targ = torch.cat([batch_obs_curr, batch_obs_next_targ], 0)

        states_local_curr_next_targ = self.network_policy.binder.forward_single(batch_obs_curr_next_targ)
        state_local_curr, state_local_next, state_local_targ = states_local_curr_next_targ.chunk(3, dim=0)
        states_local_curr_targ = torch.cat([state_local_curr, state_local_targ], -1)

        # predicted_Q = self.network_policy.estimator_Q(states_local_curr_targ, batch_action, scalarize=False)
        predicted_distance = self.network_policy.estimator_distance(states_local_curr_targ, batch_action, scalarize=False)

        with torch.no_grad():
            states_local_next_targ = torch.cat([state_local_next.detach(), state_local_targ.detach()], -1).detach()
            # predicted_Q_next = self.network_policy.estimator_Q(states_local_next_targ, scalarize=True)
            predicted_distance_next = self.network_policy.estimator_distance(states_local_next_targ, scalarize=True)
            # action_next = torch.argmax(predicted_Q_next.detach(), dim=1, keepdim=True)
            action_next = torch.argmin(predicted_distance_next.detach(), dim=1, keepdim=True)
            states_local_next_targ_targetnet = self.network_target.binder.forward_train(batch_obs_next, batch_obs_targ)
        # discount head
        with torch.no_grad():
            distance_next = self.network_target.estimator_distance(states_local_next_targ_targetnet, action_next, scalarize=True).reshape(size_batch, -1)
            distance_next[batch_done] = 1000.0
            distance_next[batch_targ_reached] = 0.0
            target_distance = 1.0 + distance_next
            target_distance_dist = self.network_target.estimator_distance.histogram_converter.to_histogram(target_distance.detach())
        distance_logits_curr = predicted_distance.reshape(size_batch, -1)
        loss_distance = torch.nn.functional.kl_div(torch.log_softmax(distance_logits_curr, -1), target_distance_dist.detach(), reduction="none").sum(-1)

        # omega head
        predicted_omega = self.network_policy.estimator_omega(state_local_next, scalarize=False)
        omega_logits_pred = predicted_omega.reshape(-1, 2)
        loss_omega = torch.nn.functional.cross_entropy(torch.log_softmax(omega_logits_pred, -1), batch_done.to(torch.long).detach(), reduction="none")

        # priorities = loss_TD.squeeze() + loss_distance.squeeze() + loss_omega.squeeze()
        priorities = loss_distance.squeeze() + loss_omega.squeeze()
        if debug:
            with torch.no_grad():
                distance_curr = self.network_policy.estimator_distance.histogram_converter.from_histogram(predicted_distance.detach())
                deviation = torch.abs(
                    distance_curr.clamp(
                        self.network_policy.estimator_distance.histogram_converter.value_min,
                        self.network_policy.estimator_distance.histogram_converter.value_max,
                    )
                    - target_distance.squeeze().clamp(
                        self.network_policy.estimator_distance.histogram_converter.value_min,
                        self.network_policy.estimator_distance.histogram_converter.value_max,
                    )
                ).detach()
                if batch_targ_reached.any():
                    deviation_targ_reached = deviation[batch_targ_reached]
                if mask_dead.any():
                    deviation_dead = deviation[mask_dead]
                writer.add_scalar("Debug/deviation", deviation.mean(), self.steps_processed)
                if batch_targ_reached.any():
                    writer.add_scalar("Debug/deviation_targ_reached", deviation_targ_reached.mean(), self.steps_processed)
                if mask_dead.any():
                    writer.add_scalar("Debug/deviation_dead", deviation_dead.mean(), self.steps_processed)
                omega_pred = omega_logits_pred.argmax(-1).bool()
                acc_omega = (batch_done == omega_pred).sum() / batch_done.shape[0]
                writer.add_scalar("Debug/acc_omega", acc_omega.item(), self.steps_processed)
        # return priorities, loss_TD, loss_distance, loss_omega, states_local_curr_targ
        return priorities, loss_distance, loss_omega, states_local_curr_targ

    @torch.no_grad()
    def decide(self, obs_curr, epsilon=None, eval=False, env=None, writer=None, random_walk=False, step_record=None):
        if epsilon is None:
            epsilon = self.epsilon_eval if eval else self.schedule_epsilon.value(self.steps_interact)
        else:
            assert epsilon >= 0 and epsilon <= 1.0
        random_walk = random_walk and not eval
        if np.random.rand() < epsilon or (random_walk and not self.episode_for_debug):
            if self.episode_for_debug:
                print(f"[step {self.steps_episode}]: random action")
            self.steps_episode += 1
            return self.action_space.sample()
        obs_curr_tensor = self.obs2tensor(obs_curr)
        if self.obs_goal_tensor is None:
            self.obs_goal_tensor = self.obs2tensor(env.obs_goal)
        if self.obses_intermediate_subgoals is None:
            self.replan = True
            if self.episode_for_debug:
                assert env is not None
                print(f"[step {self.steps_episode}]: replan due to nonexistent plan")
        else:
            coincidence = (obs_curr_tensor == self.obses_intermediate_subgoals).reshape(self.obses_intermediate_subgoals.shape[0], -1).all(-1)
            indices_coincide = torch.where(coincidence)[0]
            if len(indices_coincide):
                index_coincide = int(indices_coincide[-1])
                self.num_subgoals_finished = index_coincide + 1
                self.replan = True
                if self.episode_for_debug:
                    print(f"[step {self.steps_episode}]: replan due to subgoal reached at {env.obs2ijd(self.obses_intermediate_subgoals[index_coincide])}")
        if self.replan:
            pass
        elif self.steps_episode - self.step_last_planned >= self.freq_plan:
            self.replan = True
            self.num_planning_triggered_timeout += 1
            if self.episode_for_debug:
                print(f"[step {self.steps_episode}]: replan due to timeout ({self.freq_plan})")
        if self.replan:
            self.num_planning_triggered += 1
            self.steps_this_subgoal = 0
            if self.episode_for_debug:
                print(f"[step {self.steps_episode}]: planning triggered for the {self.num_planning_triggered}th time")
            self.replan = False
            if self.episode_for_debug:
                print(f"[step {self.steps_episode}]: step_last_planned is {self.step_last_planned}, overwrite with {self.steps_episode}")
            self.step_last_planned = self.steps_episode
            if self.code_goal is None:
                if self.episode_for_debug:
                    print(f"[step {self.steps_episode}]: self.code_goal is None, encoded")
                self.code_goal = self.cvae.encode_from_obs(self.obs_goal_tensor).reshape(-1, self.cvae.num_categoricals * self.cvae.num_categories).float()
            if self.episode_for_debug:
                print(f"[step {self.steps_episode}]: finished {self.num_subgoals_finished} subgoals")
            if self.obses_intermediate_subgoals is None:  # construct path nodes
                num_intermediate_subgoals = self.num_waypoints - 2
                if self.episode_for_debug:
                    print(f"[step {self.steps_episode}]: {num_intermediate_subgoals} intermediate_subgoals")
                size_batch_optim = 128
                prob_mask_uniform = (
                    torch.ones(num_intermediate_subgoals, self.cvae.num_categoricals, self.cvae.num_categories, device=self.device) / self.cvae.num_categories
                )
                prob_mask = prob_mask_uniform.clone()
                states_local_curr_goal = self.network_policy.binder.forward_single(torch.cat([obs_curr_tensor, self.obs_goal_tensor], dim=0))
                state_local_curr, state_local_goal = states_local_curr_goal.chunk(2, dim=0)

                num_iters = 5
                for iter in range(num_iters):
                    code_intermediate_subgoals = torch.distributions.OneHotCategorical(probs=prob_mask).sample([size_batch_optim])
                    if code_intermediate_subgoals.shape[0] == 0:
                        self.replan = True
                        if self.episode_for_debug:
                            print(f"[step {self.steps_episode}]: planning failed because no valid plan generated")
                        return self.action_space.sample()
                    size_batch_actual = code_intermediate_subgoals.shape[0]
                    code_intermediate_subgoals = code_intermediate_subgoals.reshape(
                        size_batch_actual, num_intermediate_subgoals, self.network_policy.binder.len_code
                    )

                    obses_intermediate_subgoals = self.decoder_wp(
                        code_intermediate_subgoals.reshape(-1, code_intermediate_subgoals.shape[-1]),
                        torch.repeat_interleave(obs_curr_tensor, size_batch_actual * num_intermediate_subgoals, dim=0),
                    ).to(torch.uint8)

                    states_local_intermediate = self.network_policy.binder.forward_single(obses_intermediate_subgoals)
                    states_local_intermediate = states_local_intermediate.reshape(size_batch_actual, num_intermediate_subgoals, -1)
                    states_local_start = torch.cat([state_local_curr[None, :, :].repeat_interleave(size_batch_actual, 0), states_local_intermediate], dim=1)
                    states_local_end = torch.cat([states_local_intermediate, state_local_goal[None, :, :].repeat_interleave(size_batch_actual, 0)], dim=1)
                    states_local_start_end = torch.cat([states_local_start, states_local_end], dim=-1).reshape(-1, 2 * states_local_start.shape[-1])
                    obses_intermediate_subgoals = obses_intermediate_subgoals.reshape(size_batch_actual, num_intermediate_subgoals, *obs_curr_tensor.shape[-3:])

                    omega_start = (
                        self.network_policy.estimator_omega(states_local_start.reshape(-1, states_local_start.shape[-1]), scalarize=False)
                        .argmax(-1)
                        .bool()
                        .reshape(size_batch_actual, num_intermediate_subgoals + 1)
                    )
                    # actions = self.network_policy.estimator_Q(states_local_start_end, scalarize=True).argmax(-1)
                    # vec_fit = self.network_policy.estimator_distance(states_local_start_end, actions, scalarize=True).reshape(
                    #     size_batch_actual, num_intermediate_subgoals + 1
                    # )
                    vec_fit = (
                        self.network_policy.estimator_distance(states_local_start_end, scalarize=True)
                        .min(-1)[0]
                        .reshape(size_batch_actual, num_intermediate_subgoals + 1)
                    )
                    vec_fit[omega_start] = 100
                    fitness = torch.linalg.vector_norm(vec_fit, ord=torch.inf, dim=-1)
                    fitness_elite, indices_elite = torch.topk(fitness, 16, sorted=True, largest=False)
                    elites = code_intermediate_subgoals[indices_elite].reshape(fitness_elite.shape[0], -1)
                    if fitness_elite[0] < 1.01:
                        break
                    if iter != num_iters - 1:
                        prob_mask = 0.01 * prob_mask_uniform + 0.99 * torch.mean(
                            elites.reshape(-1, num_intermediate_subgoals, self.cvae.num_categoricals, self.cvae.num_categories), 0
                        )
                if self.episode_for_debug:
                    print(f"best planning loss at last step ({iter}): {fitness_elite[0].item()}, with mean {fitness.mean().item()}")
                best = elites[[0]].reshape(num_intermediate_subgoals, self.network_policy.binder.len_code)
                self.dist_between_subgoals = vec_fit[indices_elite[0]][1:]
                self.obses_intermediate_subgoals = obses_intermediate_subgoals[indices_elite[0]]
                self.obs_targ = self.obses_intermediate_subgoals[[0]]
                if self.episode_for_debug:
                    code_subgoals = torch.cat([best, self.code_goal], 0)
                    obs_subgoals = torch.cat([self.obses_intermediate_subgoals, self.obs_goal_tensor], 0)
                    i_goals, j_goals = env.obs2ijd(obs_subgoals)
                    ijd_goals = np.stack([i_goals, j_goals], -1)
                    states_planned = [env.ijd2state(*env.agent_pos)] + env.ijd2state(i_goals, j_goals).tolist()
                    ###########################################################
                    dists_true = []
                    if env.DP_info["lava_map"] is None:
                        env.init_DP_assets()
                    if env.DP_info["P"] is None:
                        env.collect_transition_probs()
                    if env.DP_info["A"] is None:
                        env.collect_state_adjacency()
                    for idx_state_start in range(len(states_planned) - 1):
                        state_start = states_planned[idx_state_start]
                        ret = dijkstra(env.DP_info["A"], state_start)
                        dists_true.append(ret[states_planned[idx_state_start + 1]])
                    ###########################################################
                    states_local_start_end, states_local_start = self.network_policy.binder.forward_train(
                        torch.cat([obs_curr_tensor, self.obses_intermediate_subgoals], 0).detach(), obs_subgoals.detach(), return_curr=True
                    )
                    omega_start = self.network_policy.estimator_omega(states_local_start, scalarize=False).argmax(-1).bool().reshape(-1)
                    # actions = self.network_policy.estimator_Q(states_local_start_end, scalarize=True).argmax(-1)
                    # vec_fit_best = self.network_policy.estimator_distance(states_local_start_end, actions, scalarize=True).reshape(-1)
                    vec_fit_best = self.network_policy.estimator_distance(states_local_start_end, scalarize=True).min(-1)[0].reshape(-1)
                    vec_fit_best[omega_start] = 100
                    dists_estim = vec_fit_best.detach().cpu().numpy()
                    print(f"[step {self.steps_episode}]: ijd_curr {env.obs2ijd(obs_curr)}", end="")
                    list_ijd_goals = ijd_goals.tolist()
                    for i in range(len(list_ijd_goals)):
                        print(f" --{dists_estim[i]:.1f}({dists_true[i]})-->  {list_ijd_goals[i]}", end="")
                        if i == len(list_ijd_goals) - 1:
                            print("")
                    if writer is not None:
                        diff_dists_mean = np.mean(np.abs(np.clip(dists_true, 0, 100) - dists_estim))
                        if np.isinf(diff_dists_mean):
                            diff_dists_mean = np.nan
                        writer.add_scalar("LEAP_Plan_train/diff_dists", diff_dists_mean, self.steps_interact)
                        for idx_dist in range(len(dists_true)):
                            diff_dist = np.abs(np.clip(dists_true[idx_dist], 0, 100) - dists_estim[idx_dist])
                            if np.isinf(diff_dist):
                                diff_dist = np.nan
                            writer.add_scalar(f"LEAP_Plan_train/diff_dist_{idx_dist}", diff_dist, self.steps_interact)
                        code_subgoals_recon = self.cvae.encode_from_obs(obs_subgoals).reshape(-1, self.cvae.num_categoricals * self.cvae.num_categories)
                        writer.add_scalar(
                            "LEAP_Plan_train/deviation_code",
                            1.0 - (code_subgoals_recon == code_subgoals).all(-1).float().mean().item(),
                            self.steps_interact,
                        )
            else:
                states_local_start_end, states_local_start = self.network_policy.binder.forward_train(
                    obs_curr_tensor.repeat(self.num_waypoints - 1, 1, 1, 1),
                    torch.cat([self.obses_intermediate_subgoals, self.obs_goal_tensor], 0).detach(),
                    return_curr=True,
                )
                omega_start = self.network_policy.estimator_omega(states_local_start, scalarize=False).argmax(-1).bool().reshape(-1)
                # actions = self.network_policy.estimator_Q(states_local_start_end, scalarize=True).argmax(-1)
                # vec_fit_best = self.network_policy.estimator_distance(states_local_start_end, actions, scalarize=True).reshape(-1)
                vec_fit_best = self.network_policy.estimator_distance(states_local_start_end, scalarize=True).min(-1)[0].reshape(-1)
                vec_fit_best[omega_start] = 100
                dists_total = vec_fit_best.clone()
                for idx_subgoal in range(self.num_waypoints - 2):
                    dists_total[idx_subgoal] += self.dist_between_subgoals[idx_subgoal:].sum()
                idx_subgoal = dists_total.argmin(-1).item()
                if idx_subgoal == self.num_waypoints - 2:
                    self.obs_targ = self.obs_goal_tensor
                    if self.episode_for_debug:
                        print(f"[step {self.steps_episode}]: self.obs_targ set w/ target {env.obs2ijd(self.obs_targ)} (goal)")
                else:
                    self.obs_targ = self.obses_intermediate_subgoals[[[idx_subgoal]]]
                    if self.episode_for_debug:
                        print(f"[step {self.steps_episode}]: self.obs_targ set w/ target {env.obs2ijd(self.obs_targ)} ({idx_subgoal}th subgoal)")

        assert self.obs_targ is not None
        states_local_curr_targ = self.network_policy.binder.forward_train(obs_curr_tensor, self.obs_targ)
        self.steps_this_subgoal += 1
        if self.episode_for_debug:
            print(f"[step {self.steps_episode}]: this subgoal consumed {self.steps_this_subgoal} steps as of now")
        if self.steps_this_subgoal == self.steps_per_subgoal:
            self.replan = True
            if self.episode_for_debug:
                print(f"[step {self.steps_episode}]: replan due to steps_this_subgoal == steps_per_subgoal")
        # action = self.network_policy.estimator_Q(states_local_curr_targ, scalarize=True).argmax().item()
        action = self.network_policy.estimator_distance(states_local_curr_targ, scalarize=True).argmin().item()
        self.steps_episode += 1
        if random_walk:
            action = env.action_space.sample()
        return action

    def step(self, obs_curr, action, reward, obs_next, done, writer=None, add_to_buffer=True, increment_steps=True):
        if increment_steps:
            self.steps_interact += 1
        if add_to_buffer and obs_next is not None:
            sample = {"obs": np.array(obs_curr), "act": action, "rew": reward, "done": done, "next_obs": np.array(obs_next)}
            self.add_to_buffer(sample)


class LEAP(LEAP_BASE):
    def __init__(
        self,
        env,
        network_policy,
        network_target=None,
        freq_plan=4,
        num_waypoints=5,
        dist_cutoff=8,
        gamma=0.99,
        clip_reward=True,
        exploration_fraction=0.02,
        epsilon_final_train=0.01,
        epsilon_eval=0.001,
        steps_total=50000000,
        prioritized_replay=True,
        type_optimizer="Adam",
        lr=5e-4,
        eps=1.5e-4,
        time_learning_starts=20000,
        freq_targetsync=8000,
        freq_train=4,
        size_batch=64,
        func_obs2tensor=minigridobs2tensor,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        seed=42,
        hrb=None,
        silent=False,
        suppress_delusion=False,
    ):
        super(LEAP, self).__init__(
            env,
            network_policy,
            freq_plan=freq_plan,
            num_waypoints=num_waypoints,
            dist_cutoff=dist_cutoff,
            gamma=gamma,
            clip_reward=clip_reward,
            exploration_fraction=exploration_fraction,
            epsilon_final_train=epsilon_final_train,
            epsilon_eval=epsilon_eval,
            steps_total=steps_total,
            prioritized_replay=prioritized_replay,
            func_obs2tensor=func_obs2tensor,
            device=device,
            seed=seed,
            hrb=hrb,
            silent=silent,
            suppress_delusion=suppress_delusion,
        )

        self.optimizer = eval("torch.optim.%s" % type_optimizer)(self.network_policy.parameters(), lr=lr, eps=eps)

        # initialize target network
        if network_target is None:
            self.network_target = copy.deepcopy(self.network_policy)
        else:
            self.network_target = network_target
        if self.network_target.cvae is not None:
            self.network_target.cvae.to("cpu")
            self.network_target.cvae = None
        for param in self.network_target.parameters():
            param.requires_grad = False
        self.network_target.eval()
        for module in self.network_target.modules():
            module.eval()

        self.size_batch = size_batch
        self.time_learning_starts = time_learning_starts
        assert self.time_learning_starts >= self.size_batch
        self.freq_train = freq_train
        self.freq_targetsync = freq_targetsync
        self.steps_processed = 0
        self.step_last_targetsync = self.time_learning_starts

    def need_update(self):
        if self.steps_interact >= self.time_learning_starts:
            if self.hrb.get_stored_size() >= self.size_batch and (self.steps_interact - self.steps_processed) >= self.freq_train:
                return True
        return False

    def update_step(self, batch_processed=None, writer=None):
        if self.steps_interact >= self.time_learning_starts:
            if self.steps_interact - self.step_last_targetsync >= self.freq_targetsync:
                self.sync_parameters()
                self.step_last_targetsync += self.freq_targetsync
            if self.steps_interact - self.steps_processed >= self.freq_train:
                self.update(batch_processed=batch_processed, writer=writer)
                if self.steps_processed == 0:
                    self.steps_processed = self.time_learning_starts
                else:
                    self.steps_processed += self.freq_train

    def step(self, obs_curr, action, reward, obs_next, done, writer=None, add_to_buffer=True, increment_steps=True):
        """
        an agent step: in this step the agent does whatever it needs
        """
        super().step(obs_curr, action, reward, obs_next, done, writer=writer, add_to_buffer=add_to_buffer, increment_steps=increment_steps)
        self.update_step(writer=writer)

    # @profile
    def update(self, batch_processed=None, writer=None):
        """
        update the parameters of the DQN model using the weighted sampled Bellman error
        """
        debug = writer is not None and np.random.rand() < 0.05
        with torch.no_grad():
            if batch_processed is None:
                if self.prioritized_replay:
                    batch = self.hrb.sample(self.size_batch, beta=self.schedule_beta_sample_priorities.value(self.steps_interact))
                else:
                    batch = self.hrb.sample(self.size_batch)
                batch_processed = self.process_batch(batch, prioritized=self.prioritized_replay, with_targ=True)
            batch_obs_curr, batch_action, batch_reward, batch_obs_next, batch_done, batch_obs_targ, weights, batch_idxes = batch_processed
        # priorities, loss_TD, loss_distance, loss_omega, states_local_curr_targ = self.calculate_loss(
        priorities, loss_distance, loss_omega, states_local_curr_targ = self.calculate_loss(
            batch_obs_curr,
            batch_action,
            batch_reward,
            batch_obs_next,
            batch_done,
            batch_obs_targ,
            debug=debug,
            writer=writer,
        )
        if self.suppress_delusion:
            with torch.no_grad():
                _batch_obs_targ = self.cvae.imagine_batch_from_obs(batch_obs_curr)
            _, _loss_distance, _loss_omega, _states_local_curr_targ = self.calculate_loss(
                batch_obs_curr,
                batch_action,
                batch_reward,
                batch_obs_next,
                batch_done,
                _batch_obs_targ,
                debug=False,
                writer=None,
            )
            loss_distance_aux = _loss_distance.mean()
        # loss_overall = loss_TD + loss_distance + loss_omega
        loss_overall = loss_distance + loss_omega
        if self.prioritized_replay:
            assert weights is not None
            # kaixhin's rainbow implementation used mean()
            error_overall_weighted = (loss_overall * weights.detach()).mean()

        else:
            error_overall_weighted = loss_overall.mean()

        if self.suppress_delusion:
            error_overall_weighted += 0.25 * loss_distance_aux

        self.optimizer.zero_grad(set_to_none=True)
        error_overall_weighted.backward()

        if debug:
            with torch.no_grad():
                grads = [param.grad.detach().flatten() for param in self.network_policy.parameters()]
                norm_grad = torch.cat(grads).norm().item()
        torch.nn.utils.clip_grad_value_(self.network_policy.parameters(), 1.0)
        self.optimizer.step()

        with torch.no_grad():
            # update prioritized replay, if used
            if self.prioritized_replay:
                self.hrb.update_priorities(batch_idxes, priorities.detach().cpu().numpy())
                if debug:
                    writer.add_scalar("Train/priorities", priorities.mean().item(), self.steps_processed)

            if debug:
                writer.add_scalar("Debug/norm_rep_local", torch.sqrt((states_local_curr_targ**2).sum(-1)).mean().item(), self.steps_processed)
                writer.add_scalar("Debug/norm_grad", norm_grad, self.steps_processed)
                writer.add_scalar("Train/loss_distance", loss_distance.mean().item(), self.steps_processed)
                writer.add_scalar("Train/loss_omega", loss_omega.mean().item(), self.steps_processed)

    def sync_parameters(self):
        """
        synchronize the parameters of self.network_policy and self.network_target
        this is hard sync, maybe a softer version is going to do better
        cvae not synced, since we don't need it for target network
        """
        self.network_target.binder.load_state_dict(self.network_policy.binder.state_dict())
        self.network_target.estimator_distance.load_state_dict(self.network_policy.estimator_distance.state_dict())
        self.network_target.estimator_omega.load_state_dict(self.network_policy.estimator_omega.state_dict())
        if not self.silent:
            print("policy-target parameters synced")


def create_leap_network(args, env, dim_embed, num_actions, device, share_memory=False):
    if args.activation == "relu":
        activation = torch.nn.ReLU
    elif args.activation == "elu":
        activation = torch.nn.ELU
    elif args.activation == "leakyrelu":
        activation = torch.nn.LeakyReLU
    elif args.activation == "silu":
        activation = torch.nn.SiLU

    from models import Encoder_MiniGrid_Separate, Decoder_MiniGrid_Separate

    encoder_CVAE = Encoder_MiniGrid_Separate()
    decoder_CVAE = Decoder_MiniGrid_Separate()
    beta = 0.00025
    interval_beta = 10000
    obs_sample = minigridobs2tensor(env.reset())
    checkpoint = torch.load(args.path_pretrained_vae)
    cvae = CVAE_MiniGrid_Separate2(
        encoder_CVAE,
        decoder_CVAE,
        obs_sample,
        num_categoricals=checkpoint["num_categoricals"],
        num_categories=checkpoint["num_categories"],
        beta=beta,
        activation=activation,
        interval_beta=interval_beta,
    )
    cvae.load_state_dict(checkpoint["model_state_dict"])
    cvae.to(device)
    if share_memory:
        cvae.share_memory()

    from models import Predictor_MiniGrid

    binder = Binder_LEAP(
        cvae=cvae,
        len_code=checkpoint["num_categoricals"] * checkpoint["num_categories"],
        len_rep=args.len_rep,
        size_input=obs_sample.shape[-2],
        activation=activation,
    )

    binder.to(device)
    if share_memory:
        binder.share_memory()

    dict_head_Q = {"len_predict": num_actions, "dist_out": True, "value_min": 0, "value_max": 1, "atoms": args.atoms_value, "classify": False}

    # estimator_Q = Predictor_MiniGrid(
    #     num_actions,
    #     len_input=binder.len_out,
    #     depth=args.depth_hidden,
    #     width=args.width_hidden,
    #     norm=bool(args.layernorm),
    #     activation=activation,
    #     dict_head=dict_head_Q,
    # )
    # estimator_Q.to(device)
    # if share_memory:
    #     estimator_Q.share_memory()

    dict_head_distance = {
        "len_predict": num_actions,
        "dist_out": True,
        "value_min": 1,
        "value_max": args.atoms_discount,
        "atoms": args.atoms_discount,
        "classify": False,
    }
    estimator_distance = Predictor_MiniGrid(
        num_actions,
        len_input=binder.len_out,
        depth=args.depth_hidden,
        width=args.width_hidden,
        norm=bool(args.layernorm),
        activation=activation,
        dict_head=dict_head_distance,
    )
    estimator_distance.to(device)
    if share_memory:
        estimator_distance.share_memory()

    dict_head_omega = {"len_predict": 1, "dist_out": True, "value_min": 0.0, "value_max": 1.0, "atoms": 2, "classify": True}
    estimator_omega = Predictor_MiniGrid(
        num_actions,
        len_input=args.len_rep,
        depth=args.depth_hidden,
        width=args.width_hidden,
        norm=bool(args.layernorm),
        activation=activation,
        dict_head=dict_head_omega,
    )
    estimator_omega.to(device)
    if share_memory:
        estimator_omega.share_memory()

    # network_policy = LEAP_NETWORK(binder, estimator_Q, estimator_distance, estimator_omega, cvae)
    network_policy = LEAP_NETWORK(binder, estimator_distance, estimator_omega, cvae)

    if share_memory:
        network_policy.share_memory()
    return network_policy


def create_leap_agent(args, env, dim_embed, num_actions, device=None, hrb=None, network_policy=None, network_target=None, inference_only=False, silent=False):
    if device is None:
        if torch.cuda.is_available() and not args.force_cpu:
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
            warnings.warn("agent created on cpu")

    if not inference_only and hrb is None:
        from utils import get_cpprb

        hrb = get_cpprb(
            env,
            args.size_buffer,
            prioritized=args.prioritized_replay,
            hindsight=True,
            hindsight_strategy=args.hindsight_strategy,
        )

    network_policy = create_leap_network(args, env, dim_embed, num_actions, device=device, share_memory=False)

    if inference_only:
        agent = LEAP_BASE(
            env,
            network_policy,
            freq_plan=args.freq_plan,
            num_waypoints=args.num_waypoints,
            gamma=args.gamma,
            steps_total=args.steps_max,
            prioritized_replay=bool(args.prioritized_replay),
            device=device,
            seed=args.seed,
            silent=silent,
            hrb=hrb,
            suppress_delusion=args.suppress_delusion,
        )
    else:
        agent = LEAP(
            env,
            network_policy,
            freq_plan=args.freq_plan,
            num_waypoints=args.num_waypoints,
            gamma=args.gamma,
            steps_total=args.steps_max,
            prioritized_replay=bool(args.prioritized_replay),
            freq_train=args.freq_train,
            freq_targetsync=args.freq_targetsync,
            lr=args.lr,
            size_batch=args.size_batch,
            device=device,
            seed=args.seed,
            silent=silent,
            hrb=hrb,
            suppress_delusion=args.suppress_delusion,
        )
    return agent
