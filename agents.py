import torch, numpy as np, copy
import warnings
from utils import (
    LinearSchedule,
    minigridobs2tensor,
    RL_AGENT,
    abstract_planning,
    generate_random_waypoints,
    append_GT_graph,
    k_medoids,
    find_unique,
    process_batch,
)
from visual_utils import visualize_waypoint_graph, visualize_plan


class SKIPPER_NETWORK(torch.nn.Module):
    def __init__(self, encoder, binder, estimator_Q, estimator_discount, estimator_reward, estimator_omega, cvae=None):
        super(SKIPPER_NETWORK, self).__init__()
        self.encoder = encoder
        self.binder = binder
        self.estimator_Q = estimator_Q
        self.estimator_discount = estimator_discount
        self.estimator_reward = estimator_reward
        self.estimator_omega = estimator_omega
        self.cvae = cvae

    def to(self, device):
        super().to(device)
        self.encoder.to(device)
        self.binder.to(device)
        if self.estimator_Q is not None:
            self.estimator_Q.to(device)
        self.estimator_discount.to(device)
        self.estimator_reward.to(device)
        self.estimator_omega.to(device)
        if self.cvae is not None:
            self.cvae.to(device)

    def parameters(self):
        parameters = []
        parameters += list(self.encoder.parameters())
        parameters += list(self.binder.parameters())
        if self.estimator_Q is not None:
            parameters += list(self.estimator_Q.parameters())
        parameters += list(self.estimator_discount.parameters())
        parameters += list(self.estimator_reward.parameters())
        parameters += list(self.estimator_omega.parameters())
        if self.cvae is not None:
            parameters += list(self.cvae.parameters())
        return parameters


class SKIPPER_BASE(RL_AGENT):
    def __init__(
        self,
        env,
        network_policy,
        freq_plan=16,
        num_waypoints=16,
        waypoint_strategy="once",
        always_select_goal=False,
        optimal_plan=False,
        optimal_policy=False,
        dist_cutoff=8,
        prune_with_oracle=False,
        gamma=0.99,
        gamma_int=0.95,
        type_intrinsic_reward="sparse",
        clip_reward=True,
        exploration_fraction=0.02,
        epsilon_final_train=0.01,
        epsilon_eval=0.001,
        steps_total=50000000,
        prioritized_replay=True,
        func_obs2tensor=minigridobs2tensor,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        seed=42,
        valid_waypoints_only=False,
        no_lava_waypoints=False,
        hrb=None,
        silent=False,
        transform_discount_target=True,
        num_waypoints_unpruned=32,
        suppress_delusion=False,
        no_Q_head=False,
        unique_codes=False,
        unique_obses=True,
    ):
        super(SKIPPER_BASE, self).__init__(env, gamma, seed)

        self.clip_reward = clip_reward
        self.schedule_epsilon = LinearSchedule(
            schedule_timesteps=int(exploration_fraction * steps_total),
            initial_p=1.0,
            final_p=epsilon_final_train,
        )
        self.epsilon_eval = epsilon_eval

        self.gamma_int = gamma_int
        self.type_intrinsic_reward = type_intrinsic_reward

        self.device = device
        self.always_select_goal = bool(always_select_goal)
        self.optimal_plan = bool(optimal_plan)
        self.optimal_policy = bool(optimal_policy)

        self.freq_plan, self.step_last_planned = freq_plan, 0
        self.num_waypoints = num_waypoints
        assert waypoint_strategy in ["once", "regenerate_whole_graph", "grow"]
        self.waypoint_strategy = waypoint_strategy
        self.prune_with_oracle = bool(prune_with_oracle)
        self.num_waypoints_unpruned = num_waypoints_unpruned
        assert self.num_waypoints_unpruned >= self.num_waypoints

        self.network_policy = network_policy
        self.network_target = self.network_policy

        self.support_discount = self.network_policy.estimator_discount.histogram_converter.support_discount
        self.support_distance = self.network_policy.estimator_discount.histogram_converter.support_distance
        self.support_reward = self.network_policy.estimator_reward.histogram_converter.support
        self.cvae = self.network_policy.cvae

        # if self.optimal_policy:
        #     assert self.cvae is None or self.optimal_plan, "no optimal policy for non-existing states"

        if self.cvae is None:
            self.encoder_wp = lambda obs, env: np.array(env.obs2ijd(obs))
            self.decoder_wp = lambda ijd, env: env.ijd2obs(*ijd)
        else:
            self.encoder_wp = lambda obs: self.cvae.encode_from_obs(obs).reshape(obs.shape[0], -1).squeeze_().cpu().numpy()
            self.decoder_wp = lambda code, obs: self.cvae.decode_to_obs(code, obs)
        self.suppress_delusion = bool(suppress_delusion)

        self.valid_waypoints_only = bool(valid_waypoints_only)
        self.no_lava_waypoints = bool(no_lava_waypoints)

        self.transform_discount_target = bool(transform_discount_target)
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

        self.no_Q_head = bool(no_Q_head)
        self.unique_codes = bool(unique_codes)
        self.unique_obses = bool(unique_obses)

        self.on_episode_end(eval=True)  # NOTE: do not call hrb.on_episode_end() here when there is no experience

    def add_to_buffer(self, batch):
        self.hrb.add(**batch)

    @torch.no_grad()
    def process_batch(self, batch, prioritized=False, with_targ=False):
        return process_batch(
            batch, prioritized=prioritized, with_targ=with_targ, device=self.device, obs2tensor=minigridobs2tensor, clip_reward=self.clip_reward, aux=False
        )

    @torch.no_grad()
    def another_waypoint_reached(self, obs_curr, env):
        if self.waypoints_existing is None:
            return False
        if self.waypoint_curr is None:
            if self.cvae is None:
                self.waypoint_curr = self.encoder_wp(obs_curr, env)
            else:
                self.waypoint_curr = self.encoder_wp(self.obs2tensor(obs_curr))
        if self.waypoint_targ is not None:
            if (self.waypoint_curr == self.waypoint_targ).all():
                self.waypoint_last_reached = copy.copy(self.waypoint_curr)
                self.idx_wp_last_reached = int(self.idx_waypoint_targ)
                self.num_waypoints_reached += 1
                if not self.silent:
                    print(f"planning triggered at step {self.steps_interact:d}: waypoint_targ {self.waypoint_targ.tolist()} reached")
                self.waypoint_targ, self.state_wp_targ, self.idx_waypoint_targ = None, None, None
                return True
        coincidence = (self.waypoints_existing == self.waypoint_curr).all(-1)
        if self.waypoint_last_reached is not None:
            coincidence &= (self.waypoints_existing != self.waypoint_last_reached).any(-1)
        found = coincidence.any()
        if found:
            self.waypoint_last_reached = copy.copy(self.waypoint_curr)
            self.idx_wp_last_reached = np.where(coincidence)[0][0]
            self.num_waypoints_reached += 1
            if not self.silent:
                print(
                    f"planning triggered at step {self.steps_interact:d}: unexpected waypoint {self.waypoint_curr.tolist()} reached",
                    end="\n" if self.waypoint_targ is None else "",
                )
                if self.waypoint_targ is not None:
                    print(f", instead of {self.waypoint_targ.tolist()}")
        return found

    def Q_conditioned(self, batch_curr, waypoint_targ=None, type_curr="obs", env=None, obs_targ=None):  # used in evaluate_multihead
        """
        fast forward pass for conditioned Q
        """
        assert waypoint_targ is not None or obs_targ is not None
        if obs_targ is None:
            if self.cvae is None:
                if self.obs_wp_targ is None:
                    self.obs_wp_targ = self.obs2tensor(self.decoder_wp(waypoint_targ, env))
                obs_targ = self.obs_wp_targ
            else:
                obs_targ = self.obs2tensor(self.decoder_wp(waypoint_targ, env))
        elif isinstance(obs_targ, np.ndarray):
            obs_targ = self.obs2tensor(obs_targ)
        state_targ = self.network_policy.encoder(obs_targ)
        if type_curr == "obs":
            if isinstance(batch_curr, np.ndarray):
                batch_obs_curr = self.obs2tensor(batch_curr)
            else:
                batch_obs_curr = batch_curr
            state_curr = self.network_policy.encoder(batch_obs_curr)
        elif type_curr == "state_rep":
            state_curr = batch_curr
        if state_curr.shape[0] > 1 and state_targ.shape[0] == 1:
            state_targ = state_targ.expand_as(state_curr)
        state_local_binded = self.network_policy.binder(state_curr, state_targ)
        if self.no_Q_head:
            dist_discounts = self.network_policy.estimator_discount(state_local_binded, scalarize=False).softmax(-1)
            return dist_discounts @ self.support_discount
        else:
            return self.network_policy.estimator_Q(state_local_binded, scalarize=True)

    @torch.no_grad()
    def reinit_plan(self):
        self.waypoint_last_reached = None
        self.idx_wp_last_reached = None
        self.idx_waypoint_targ = None
        self.waypoint_targ = None
        self.state_wp_targ = None
        self.replan = True

    @torch.no_grad()
    def on_episode_end(self, eval=False):
        if self.optimal_policy:
            self.Q_oracle, self.pos_goal_oracle = None, None
        self.reinit_plan()
        self.waypoints_existing = None
        self.wp_existing_obses = None
        self.wp_coincidence = None
        self.replan = True
        if self.wp_graph_curr is not None:
            del self.wp_graph_curr
            self.wp_graph_curr = None
        self.num_planning_triggered = 0
        self.num_planning_triggered_timeout = 0
        self.num_waypoints_reached = 0
        self.code_goal = None
        if self.hrb is not None and not eval:
            self.hrb.on_episode_end()

    # @profile
    def calculate_multihead_error(
        self,
        batch_obs_curr,
        batch_action,
        batch_reward,
        batch_obs_next,
        batch_done,
        batch_obs_targ,
        batch_reward_int=None,
        calculate_Q_error=True,
        calculate_reward_error=True,
        calculate_omega_error=True,
        calculate_priorities=True,
        freeze_encoder=False,
        freeze_binder=False,
        type_priorities="kl",  # "kanto"
    ):
        size_batch = batch_obs_curr.shape[0]

        with torch.no_grad():
            batch_targ_reached = (batch_obs_next == batch_obs_targ).reshape(size_batch, -1).all(-1)
            # batch_didnt_move = (batch_obs_curr == batch_obs_next).reshape(size_batch, -1).all(-1)
            # batch_targ_already_reached = (batch_obs_curr == batch_obs_targ).reshape(size_batch, -1).all(-1)
            # batch_targ_already_reached_and_again = torch.logical_and(batch_targ_reached, batch_didnt_move)
            batch_done_augmented = torch.logical_or(batch_targ_reached, batch_done)
            # batch_done_augmented_except_rewarding = torch.logical_and(batch_done_augmented, batch_reward.squeeze() == 0
            batch_obs_next_targ = torch.cat([batch_obs_next, batch_obs_targ], 0)
            batch_obs_curr_next_targ = torch.cat([batch_obs_curr, batch_obs_next_targ], 0)

        with torch.set_grad_enabled(not freeze_encoder):
            batch_state_curr_next_targ = self.network_policy.encoder(batch_obs_curr_next_targ)
            batch_state_curr = batch_state_curr_next_targ[:size_batch]
            # batch_state_curr, batch_state_next, batch_state_targ = torch.split(
            #     batch_state_curr_next_targ, [size_batch, size_batch, size_batch], dim=0
            # )
        with torch.set_grad_enabled(not freeze_binder):
            if self.network_policy.binder.local_perception:
                state_local_curr_next_targ = self.network_policy.binder.extract_local_field(batch_state_curr_next_targ)
            else:
                state_local_curr_next_targ = self.network_policy.binder.flattener(batch_state_curr_next_targ)
            state_local_curr, state_local_next, state_local_targ = torch.split(state_local_curr_next_targ, [size_batch, size_batch, size_batch], dim=0)
            states_local_curr_targ = torch.cat([state_local_curr, state_local_targ], -1)

        if not self.no_Q_head:
            predicted_Q = self.network_policy.estimator_Q(states_local_curr_targ, batch_action, scalarize=False)
        predicted_discount = self.network_policy.estimator_discount(states_local_curr_targ, batch_action, scalarize=False)
        logits_reward_curr = self.network_policy.estimator_reward(states_local_curr_targ, batch_action, scalarize=False)
        # TODO(H): reuse batch_state_next_targetnet, batch_state_curr, state_local_curr, state_local_next

        with torch.no_grad():
            states_local_next_targ = torch.cat([state_local_next.detach(), state_local_targ.detach()], -1)

            if self.no_Q_head:
                softmax_predicted_discount_next = self.network_policy.estimator_discount(states_local_next_targ.detach(), scalarize=False).softmax(-1)
                predicted_discount_next = softmax_predicted_discount_next @ self.support_discount
                action_next = torch.argmax(predicted_discount_next.detach(), dim=1, keepdim=True)
            else:
                predicted_Q_next = self.network_policy.estimator_Q(states_local_next_targ.detach(), scalarize=True)
                action_next = torch.argmax(predicted_Q_next.detach(), dim=1, keepdim=True)

            batch_state_next_targ_targetnet = self.network_target.encoder(batch_obs_next_targ)
            batch_state_next_targetnet, batch_state_targ_targetnet = torch.split(batch_state_next_targ_targetnet, [size_batch, size_batch], dim=0)
            states_local_next_targ_targetnet = self.network_target.binder(batch_state_next_targetnet, batch_state_targ_targetnet)

        # discount head
        with torch.no_grad():
            dist_discounts = self.network_target.estimator_discount(states_local_next_targ_targetnet, action_next, scalarize=False).softmax(-1)
            if self.transform_discount_target:
                distance_next = (dist_discounts @ self.support_distance).reshape(size_batch, 1)
                distance_next[batch_done] = 1000.0
                distance_next[batch_targ_reached] = 0.0
                target_discount_distance = 1.0 + distance_next
                # target_discount_distance[batch_targ_already_reached_and_again] = 0.0
            else:
                discount_next = (dist_discounts @ self.network_target.estimator_discount.histogram_converter.support_discount).reshape(size_batch, 1)
                discount_next[batch_done] = 0.0
                discount_next[batch_targ_reached] = 1.0
                target_discount_distance = self.gamma * discount_next
                # target_discount_distance[batch_targ_already_reached_and_again] = 1.0
            target_discount_dist = self.network_target.estimator_discount.histogram_converter.to_histogram(target_discount_distance)
        discount_logits_curr = predicted_discount.reshape(size_batch, -1)
        loss_discount = torch.nn.functional.kl_div(torch.log_softmax(discount_logits_curr, -1), target_discount_dist.detach(), reduction="none").sum(-1)

        # Q head
        if calculate_Q_error and not self.no_Q_head:
            with torch.no_grad():
                values_next = self.network_target.estimator_Q(states_local_next_targ_targetnet, action=action_next, scalarize=True).reshape(size_batch, -1)
                if self.type_intrinsic_reward == "sparse":
                    batch_reward_int = batch_targ_reached.float().reshape(size_batch, -1) if batch_reward_int is None else batch_reward_int
                    values_next[batch_done_augmented] = 0
                elif self.type_intrinsic_reward == "dense":
                    batch_reward_int = torch.full_like(batch_reward, -1) if batch_reward_int is None else batch_reward_int
                    values_next[batch_done] = -1000
                    values_next[batch_targ_reached] = 0
                else:
                    raise NotImplementedError()
                # TODO(H): batch_reward here should be intrinsic reward, how to balance the reward-respecting perspective?
                target_Q = batch_reward_int + self.gamma_int * values_next
                Q_dist_target = self.network_target.estimator_Q.histogram_converter.to_histogram(target_Q)
            Q_logits_curr = predicted_Q.reshape(size_batch, -1)
            loss_TD = torch.nn.functional.kl_div(torch.log_softmax(Q_logits_curr, -1), Q_dist_target.detach(), reduction="none").sum(-1)
        else:
            loss_TD = torch.zeros_like(loss_discount)

        if calculate_reward_error:
            # G head
            with torch.no_grad():
                G_next = self.network_target.estimator_reward(states_local_next_targ_targetnet, action=action_next, scalarize=True).reshape(size_batch, -1)
                G_next[batch_done_augmented] = 0.0
                target_G = batch_reward + self.gamma * G_next
                G_dist_target = self.network_target.estimator_reward.histogram_converter.to_histogram(target_G)
            G_logits_curr = logits_reward_curr.reshape(size_batch, -1)
            loss_reward = torch.nn.functional.kl_div(torch.log_softmax(G_logits_curr, -1), G_dist_target.detach(), reduction="none").sum(-1)
        else:
            loss_reward = torch.zeros_like(loss_discount)
        if calculate_omega_error:  # omega head: only cross entropy
            predicted_omega = self.network_policy.estimator_omega(state_local_next, scalarize=False)
            omega_logits_pred = predicted_omega.reshape(-1, 2)
            loss_omega = torch.nn.functional.cross_entropy(torch.log_softmax(omega_logits_pred, -1), batch_done.to(torch.long).detach(), reduction="none")
        else:
            omega_logits_pred = None
            loss_omega = torch.zeros_like(loss_TD)
        ####################################################
        if calculate_priorities:
            with torch.no_grad():
                if type_priorities == "kanto":
                    kanto_discount = (target_discount_dist - discount_logits_curr.softmax(-1)).abs_().sum(-1)
                    if not calculate_reward_error:
                        kanto_reward = torch.zeros_like(kanto_discount)
                    else:
                        kanto_reward = (G_dist_target - G_logits_curr.softmax(-1)).abs_().sum(-1)
                    if not calculate_Q_error or self.no_Q_head:
                        kanto_Q = torch.zeros_like(kanto_discount)
                    else:
                        kanto_Q = (Q_dist_target - Q_logits_curr.softmax(-1)).abs_().sum(-1)
                    priorities = 0.5 * (kanto_Q + kanto_discount + kanto_reward).detach()
                elif type_priorities == "kl":
                    priorities = (loss_TD + loss_discount + loss_reward + loss_omega).squeeze().detach()  # * 0.25
                else:
                    raise NotImplementedError()
        else:
            priorities = None
        ####################################################
        return priorities, loss_TD, loss_discount, loss_reward, loss_omega, omega_logits_pred, batch_state_curr, state_local_curr

    @torch.no_grad()
    # @profile
    def get_abstract_graph(self, dict_waypoints, obs_curr=None, env=None, save_wp_existing_obses=False):
        # NOTE(H): if obs_curr is not passed, the first waypoint is not gonna be modified
        waypoints_existing = dict_waypoints["ijds"]
        if isinstance(obs_curr, np.ndarray):
            obs_curr = self.obs2tensor(obs_curr)
        if self.wp_existing_obses is None:
            if self.cvae is None:
                assert env is not None
                wp_existing_obses = self.obs2tensor(self.decoder_wp(np.split(waypoints_existing, waypoints_existing.shape[1], axis=1), env))
            else:
                wp_existing_obses = self.obs2tensor(dict_waypoints["obses"])
            if save_wp_existing_obses:
                self.wp_existing_obses = wp_existing_obses
        else:
            assert waypoints_existing.shape[0] == self.wp_existing_obses.shape[0]
            wp_existing_obses = self.wp_existing_obses
        if obs_curr is None:
            wp_obses = wp_existing_obses
        else:
            wp_obses = torch.cat([obs_curr.reshape(1, *wp_existing_obses.shape[1:]), wp_existing_obses], dim=0)
        num_waypoints = wp_obses.shape[0]
        wp_states = self.network_policy.encoder(wp_obses)
        # NOTE(H): we are exploiting the fact that binder treats two inputs independently
        if self.network_policy.binder.local_perception:
            wp_states_local = self.network_policy.binder.extract_local_field(wp_states)
        else:
            wp_states_local = self.network_policy.binder.flattener(wp_states)
        tuples = torch.cat([torch.repeat_interleave(wp_states_local, num_waypoints, dim=0), wp_states_local.repeat([num_waypoints, 1])], -1)
        omegas = self.network_policy.estimator_omega(wp_states_local, scalarize=True).bool().squeeze()
        if self.no_Q_head:
            softmax_discount_dist = self.network_policy.estimator_discount(tuples, scalarize=False).softmax(-1)
            predicted_discounts = softmax_discount_dist @ self.support_discount
            actions_greedy = torch.argmax(predicted_discounts, dim=1, keepdim=True)
            discounts = predicted_discounts.gather(1, actions_greedy).reshape(num_waypoints, num_waypoints)
            dist_discounts = softmax_discount_dist[
                torch.arange(softmax_discount_dist.shape[0], device=softmax_discount_dist.device),
                actions_greedy.squeeze(),
            ]
        else:
            predicted_Q = self.network_policy.estimator_Q(tuples, scalarize=True)
            actions_greedy = torch.argmax(predicted_Q, dim=1, keepdim=True)
            dist_discounts = self.network_policy.estimator_discount(tuples, actions_greedy, scalarize=False).softmax(-1)
            discounts = (dist_discounts @ self.support_discount).reshape(num_waypoints, num_waypoints)
        distances = (dist_discounts @ self.support_distance).reshape(num_waypoints, num_waypoints)
        rewards = self.network_policy.estimator_reward(tuples, actions_greedy, scalarize=True).reshape(num_waypoints, num_waypoints)
        return dict(discounts=discounts, distances=distances, rewards=rewards, omegas=omegas, Q=None)

    @torch.no_grad()
    def visualize_events2ijs(self, obs_curr, env, codes_all=None, writer=None, step_record=None):
        """
        generate all obses corresponding to the codes, get the ijs
        generate a list of code to ij lists
        visualize in some way
        it must be the case that now every event code is mapping to all of the possible states + potentially some impossible ones,
        therefore an argmax is preferred

        """
        if codes_all is None:
            codes_all = self.cvae.samples_uniform.reshape(self.cvae.samples_uniform.shape[0], -1)
        layout, mask_agent = self.cvae.layout_extractor(obs_curr)
        layout = layout.repeat(self.cvae.samples_uniform.shape[0], 1, 1, 1)
        obs_curr_repeated = obs_curr.repeat(self.cvae.samples_uniform.shape[0], 1, 1, 1)
        mask_agent_pred = self.cvae.forward(obs_curr_repeated, samples=codes_all, train=False)
        obs_targs = self.cvae.decoder(layout, mask_agent_pred).cpu().numpy()
        # ijds = env.obs2ijd(obs_targs)
        states, ijds = env.obs2ijdstate(obs_targs)
        ijs = np.stack(ijds, -1)
        assert self.cvae.num_categories == 2
        int_codes_all = codes_all.reshape(-1, self.cvae.num_categoricals, self.cvae.num_categories).argmax(-1).float() @ torch.flip(
            torch.pow(2, torch.arange(self.cvae.num_categoricals, device=codes_all.device, dtype=codes_all.dtype)), (0,)
        )
        int_codes_all = int_codes_all.long().cpu().numpy().tolist()
        correspondence = {}
        for idx_int_code in range(len(int_codes_all)):
            int_code = int_codes_all[idx_int_code]
            correspondence[str(int_code)] = []
        for idx_ij in range(ijs.shape[0]):
            int_code = int_codes_all[idx_ij]
            correspondence[str(int_code)].append(ijs[idx_ij].tolist())
        indices_unique_ijs = find_unique(torch.tensor(ijs, device=codes_all.device))
        # print(f"latents focus on {len(indices_unique_ijs)} states")
        writer.add_scalar("Train_CVAE/concentration_s2z", len(indices_unique_ijs), step_record)
        # TODO(H): check with eyes first then do the visualization

    def get_random_action(self, trigger_replan=True):
        if trigger_replan:
            self.replan = True
        return self.action_space.sample()

    @torch.no_grad()
    # @profile
    def decide(self, obs_curr, epsilon=None, eval=False, env=None, writer=None, random_walk=False, step_record=None):
        if epsilon is None:
            epsilon = self.epsilon_eval if eval else self.schedule_epsilon.value(self.steps_interact)
        else:
            assert epsilon >= 0 and epsilon <= 1.0
        debug = writer is not None and self.num_planning_triggered == 0 and np.random.rand() < 0.05
        if np.random.rand() < epsilon or (random_walk and not debug):
            return self.get_random_action()
        debug_visualize = debug if eval else debug and np.random.rand() < 0.1
        if debug:
            if eval:
                prefix_plan, prefix_debug, prefix_vis = "Plan_Eval", "Debug_Eval", "Visualize_Eval"
            else:
                prefix_plan, prefix_debug, prefix_vis = "Plan", "Debug", "Visualize"
            if step_record is None:
                step_record = self.steps_interact
        obs_curr_tensor = None
        generate_graph = self.waypoints_existing is None or self.waypoint_strategy == "regenerate_whole_graph"
        self.waypoint_curr = None

        if self.replan:
            pass
        elif generate_graph:
            self.replan = True
        elif self.another_waypoint_reached(obs_curr, env):
            self.replan = True
        elif self.steps_interact - self.step_last_planned >= self.freq_plan:
            self.replan = True
            self.num_planning_triggered_timeout += 1
        if self.replan:
            self.num_planning_triggered += 1
            self.replan = False
            self.step_last_planned = self.steps_interact
            # NOTE: don't generate at the start of the episode, we don't want to waste time generating the graph if plan is not even called
            if generate_graph:
                self.reinit_plan()
                self.wp_existing_obses = None
                self.wp_graph_curr = dict(omegas=None)
                if self.cvae is None:  # NOTE: using oracle
                    wp_graph_curr_true_unpruned = generate_random_waypoints(
                        env,
                        self.num_waypoints_unpruned,
                        generate_DP_info=False,
                        render=debug_visualize,
                        valid_only=self.valid_waypoints_only,
                        no_lava=self.no_lava_waypoints,
                        return_dist=self.prune_with_oracle,
                        return_obs=True,
                        unique=False,
                        obs_curr=obs_curr,
                    )
                else:
                    if obs_curr_tensor is None:
                        obs_curr_tensor = self.obs2tensor(obs_curr)
                    self.waypoint_curr = self.encoder_wp(obs_curr_tensor)
                    _, obses_pred_tensor = self.cvae.generate_from_obs(obs_curr_tensor, num_samples=self.num_waypoints_unpruned - 2)
                    obses_pred = np.concatenate([obs_curr[None, :], obses_pred_tensor.cpu().numpy(), env.obs_goal[None, :]], 0)

                    states, ijds = env.obs2ijdstate(obses_pred)
                    ijds = np.stack(ijds[: len(ijds) - int(env.ignore_dir)], 1)
                    rendered = env.render_image(ijds) if debug_visualize else None
                    wp_graph_curr_true_unpruned = {"ijds": ijds, "states": states, "obses": obses_pred, "rendered": rendered}

                    if self.prune_with_oracle:
                        raise NotImplementedError("too lazy")

                if self.prune_with_oracle:
                    # NOTE: watch out for nodes that are not reachable
                    dist = np.clip(wp_graph_curr_true_unpruned["distance"], 1, 1000)
                else:
                    # NOTE(H): the obses are generated here with oracle agents
                    wp_graph_curr_unpruned = self.get_abstract_graph(wp_graph_curr_true_unpruned, env=env, save_wp_existing_obses=True)
                wp_graph_curr_unpruned["omegas"][0] = False  # NOTE: current state is never terminal
                wp_graph_curr_unpruned["discounts"][wp_graph_curr_unpruned["omegas"]] = 0.0
                wp_graph_curr_unpruned["rewards"][wp_graph_curr_unpruned["omegas"]] = 0.0

                if self.unique_obses:
                    if self.cvae is None:
                        indices_unique_obses = find_unique(self.obs2tensor(wp_graph_curr_true_unpruned["obses"]), must_keep=[0, -1])
                    else:
                        indices_unique_obses = find_unique(torch.cat([obs_curr_tensor, obses_pred_tensor, self.obs2tensor(env.obs_goal)], 0), must_keep=[0, -1])
                if self.unique_codes:
                    indices_unique_codes = find_unique(wp_graph_curr_true_unpruned["codes"].reshape(self.num_waypoints_unpruned, -1), must_keep=[0, -1])
                if self.unique_obses and self.unique_codes:
                    indices_unique = np.intersect1d(indices_unique_obses, indices_unique_codes).tolist()
                elif self.unique_obses and not self.unique_codes:
                    indices_unique = indices_unique_obses
                elif not self.unique_obses and self.unique_codes:
                    indices_unique = indices_unique_codes
                else:
                    indices_unique = np.arange(self.num_waypoints_unpruned).tolist()

                if debug:
                    if self.unique_obses:
                        writer.add_scalar(f"{prefix_plan}/num_waypoints_unpruned_unique_obs", len(indices_unique_obses), step_record)
                    if self.unique_codes:
                        writer.add_scalar(f"{prefix_plan}/num_waypoints_unpruned_unique_code", len(indices_unique_codes), step_record)
                    writer.add_scalar(f"{prefix_plan}/num_waypoints_unpruned_unique", len(indices_unique), step_record)

                assert indices_unique[0] == 0 and indices_unique[-1] == self.num_waypoints_unpruned - 1
                if len(indices_unique) > self.num_waypoints:
                    dist = (wp_graph_curr_unpruned["distances"][indices_unique, :][:, indices_unique]).clamp_(0, 1000)
                    dist[wp_graph_curr_unpruned["omegas"][indices_unique]] = 1000
                    dist.fill_diagonal_(0)
                    dist = torch.minimum(dist, dist.T)
                    indices_chosen, _, _ = k_medoids(dist, self.num_waypoints, [0, len(indices_unique) - 1])
                    assert indices_chosen[0] == 0 and indices_chosen[-1] == dist.shape[0] - 1
                    indices_chosen = np.array(indices_unique)[indices_chosen].tolist()
                else:
                    indices_chosen = indices_unique
                indices_chosen_1p = indices_chosen[1:]
                self.wp_existing_obses = self.wp_existing_obses[indices_chosen_1p]
                if self.cvae is None:
                    self.waypoints_existing = wp_graph_curr_true_unpruned["ijds"][indices_chosen_1p]
                else:
                    self.waypoints_existing = self.encoder_wp(self.wp_existing_obses).reshape(self.wp_existing_obses.shape[0], -1)

                if self.prune_with_oracle:
                    self.wp_graph_curr.update(
                        self.get_abstract_graph(dict(ijds=self.waypoints_existing), obs_curr=obs_curr, env=env, save_wp_existing_obses=True)
                    )
                else:
                    num_waypoints_chosen = len(indices_chosen)
                    self.wp_graph_curr["ijds"] = wp_graph_curr_true_unpruned["ijds"][indices_chosen, :]
                    self.wp_graph_curr["states"] = wp_graph_curr_true_unpruned["states"][indices_chosen]
                    mask_chosen = torch.zeros_like(wp_graph_curr_unpruned["distances"], dtype=torch.int64)
                    mask_chosen[indices_chosen, :] += 1
                    mask_chosen[:, indices_chosen] += 1
                    mask_chosen = mask_chosen == 2
                    self.wp_graph_curr["distances"] = torch.masked_select(wp_graph_curr_unpruned["distances"], mask_chosen).reshape(
                        num_waypoints_chosen, num_waypoints_chosen
                    )
                    self.wp_graph_curr["discounts"] = torch.masked_select(wp_graph_curr_unpruned["discounts"], mask_chosen).reshape(
                        num_waypoints_chosen, num_waypoints_chosen
                    )
                    self.wp_graph_curr["rewards"] = torch.masked_select(wp_graph_curr_unpruned["rewards"], mask_chosen).reshape(
                        num_waypoints_chosen, num_waypoints_chosen
                    )
                    # self.wp_graph_curr["distances"] = wp_graph_curr_unpruned["distances"][indices_chosen, :][:, indices_chosen]
                    # self.wp_graph_curr["discounts"] = wp_graph_curr_unpruned["discounts"][indices_chosen, :][:, indices_chosen]
                    # self.wp_graph_curr["rewards"] = wp_graph_curr_unpruned["rewards"][indices_chosen, :][:, indices_chosen]
                    if wp_graph_curr_unpruned["omegas"] is not None:
                        self.wp_graph_curr["omegas"] = wp_graph_curr_unpruned["omegas"][indices_chosen]
                if debug_visualize:
                    img_distances = visualize_waypoint_graph(wp_graph_curr_true_unpruned["rendered"], self.wp_graph_curr, annotation="distances")
                    writer.add_image(f"{prefix_vis}/distances", img_distances, step_record, dataformats="HWC")
                    img_rewards = visualize_waypoint_graph(wp_graph_curr_true_unpruned["rendered"], self.wp_graph_curr, annotation="rewards")
                    writer.add_image(f"{prefix_vis}/rewards", img_rewards, step_record, dataformats="HWC")
                    img_discounts = visualize_waypoint_graph(wp_graph_curr_true_unpruned["rendered"], self.wp_graph_curr, annotation="discounts")
                    writer.add_image(f"{prefix_vis}/discounts", img_discounts, step_record, dataformats="HWC")
                self.wp_graph_curr["selected"] = np.zeros(len(indices_chosen), dtype=bool)
                self.waypoint_last_reached = None
            else:
                if self.waypoint_curr is None:
                    if self.cvae is None:
                        self.waypoint_curr = self.encoder_wp(obs_curr, env)
                    else:
                        self.waypoint_curr = self.encoder_wp(self.obs2tensor(obs_curr))
                aux = self.get_abstract_graph(
                    dict(ijds=self.wp_graph_curr["ijds"][1:], states=self.wp_graph_curr["states"][1:], obses=self.wp_existing_obses),
                    obs_curr=obs_curr,
                    env=env,
                    save_wp_existing_obses=False,
                )
                self.wp_graph_curr["discounts"], self.wp_graph_curr["distances"] = aux["discounts"], aux["distances"]
                self.wp_graph_curr["rewards"] = aux["rewards"]
                self.wp_graph_curr["omegas"] = aux["omegas"]

            omegas_plan = self.wp_graph_curr["omegas"]
            discounts_plan = self.wp_graph_curr["discounts"].clone()
            rewards_plan = self.wp_graph_curr["rewards"].clone()
            distances_plan = self.wp_graph_curr["distances"].clone()

            mask_cutoff = self.wp_graph_curr["distances"] > self.dist_cutoff
            mask_cutoff.fill_diagonal_(True)
            coincidence = (self.waypoints_existing == self.waypoint_curr).all(-1)
            if not self.optimal_plan and coincidence.all():  # NOTE: all waypoints coincident with the current one
                return self.get_random_action()
            if coincidence.size == 1:  # NOTE: only curr and goal
                coincidence = torch.tensor([True, False], dtype=torch.bool, device=distances_plan.device)
            else:
                coincidence = torch.tensor([True] + coincidence.tolist(), dtype=torch.bool, device=distances_plan.device)
            mask_cutoff[:, coincidence] = True
            if not self.optimal_plan and mask_cutoff[0, :].all():  # NOTE: no other waypoints is reachable from the agent
                return self.get_random_action()
            mask_cutoff[omegas_plan] = True
            discounts_plan.masked_fill_(mask_cutoff, 0.0)
            rewards_plan.masked_fill_(mask_cutoff, 0.0)
            distances_plan.masked_fill_(mask_cutoff, 1024.0)

            # NOTE(H): omega and no_loop are both covered by mask_cutoff
            Q, num_iters_plan, converged = abstract_planning(discounts_plan, rewards_plan, max_iters=5, no_loop=True)

            if self.optimal_policy:
                self.Q_oracle, self.pos_goal_oracle = None, None

            if debug_visualize and generate_graph:
                if self.cvae is not None:
                    if obs_curr_tensor is None:
                        obs_curr_tensor = self.obs2tensor(obs_curr)
                    self.visualize_events2ijs(obs_curr_tensor, env, codes_all=None, writer=writer, step_record=step_record)
                wp_graph_curr_copy = dict(
                    ijds=self.wp_graph_curr["ijds"], Q=Q, distances=distances_plan, rewards=rewards_plan, discounts=discounts_plan, omegas=omegas_plan
                )
                rendered = env.render_image(wp_graph_curr_copy["ijds"])
                img_plan = visualize_plan(rendered, wp_graph_curr_copy, Q, alpha=0.5)
                writer.add_image(f"{prefix_vis}/plan", img_plan, step_record, dataformats="HWC")
                img_Q = visualize_waypoint_graph(rendered, wp_graph_curr_copy, annotation="Q")
                writer.add_image(f"{prefix_vis}/Q", img_Q, step_record, dataformats="HWC")
                img_distances_plan = visualize_waypoint_graph(rendered, wp_graph_curr_copy, annotation="distances")
                writer.add_image(f"{prefix_vis}/distances_plan", img_distances_plan, step_record, dataformats="HWC")
                img_discounts_plan = visualize_waypoint_graph(rendered, wp_graph_curr_copy, annotation="discounts")
                writer.add_image(f"{prefix_vis}/discounts_plan", img_discounts_plan, step_record, dataformats="HWC")
                img_rewards_plan = visualize_waypoint_graph(rendered, wp_graph_curr_copy, annotation="rewards")
                writer.add_image(f"{prefix_vis}/rewards_plan", img_rewards_plan, step_record, dataformats="HWC")
            if debug:
                writer.add_scalar(f"{prefix_plan}/num_iters", int(num_iters_plan), step_record)
                writer.add_scalar(f"{prefix_plan}/VI_converged", float(converged), step_record)
                if converged:
                    num_iters_plan_converge = num_iters_plan
                else:
                    _, num_iters_plan_converge, _ = abstract_planning(discounts_plan, rewards_plan, omegas_plan, max_iters=1000)
                writer.add_scalar(f"{prefix_plan}/num_iters_converge", int(num_iters_plan_converge), step_record)

            Q_wp_curr = Q[0].cpu().numpy()
            Q_wp_curr[0] = -np.inf  # NOTE: do not target the agent location
            idx_targs = np.where(np.abs(np.max(Q_wp_curr) - Q_wp_curr) < 1e-5)[0].tolist()
            if len(idx_targs) > 1:
                distances_targs = np.take_along_axis(distances_plan[0, :].cpu().numpy(), np.array(idx_targs), -1)
                idx_targs = [idx_targs[index] for index in distances_targs.argsort().tolist()]
                assert len(idx_targs), f"distances_targs.argsort().tolist(): {distances_targs.argsort().tolist()}"
            idx_targ = int(idx_targs[0])  # NOTE(H): favor more robust targs

            if self.optimal_plan or debug:  # for debugging, fold this for better peace of mind
                ijd_curr = np.array(env.obs2ijd(obs_curr)[: 3 - int(env.ignore_dir)])
                wp_graph_GT = {}
                wp_graph_GT["ijds"] = np.concatenate([ijd_curr.reshape(1, *self.wp_graph_curr["ijds"].shape[1:]), self.wp_graph_curr["ijds"][1:]], 0)
                wp_graph_GT["states"] = np.array([env.ijd2state(*ijd_curr)] + self.wp_graph_curr["states"][1:].tolist())
                temp = append_GT_graph(env, wp_graph_GT)  # NOTE: the following ground truths include the current waypoint and the others
                discounts_GT, distances_GT = torch.tensor(temp["discount"], device=discounts_plan.device), torch.tensor(
                    temp["distance"], device=discounts_plan.device
                )
                rewards_GT, omegas_GT = torch.tensor(temp["reward"], device=discounts_plan.device), torch.tensor(temp["done"], device=discounts_plan.device)

                Q_GT, _, _ = abstract_planning(discounts_GT, rewards_GT, omegas_GT, max_iters=5)
                Q_wp_curr_GT = Q_GT[0].cpu().numpy()
                idx_targs_optimal = np.where(np.abs(np.max(Q_wp_curr_GT) - Q_wp_curr_GT) < 1e-5)[0].tolist()
                discounts_targs_optimal = np.take_along_axis(discounts_GT.cpu().numpy()[0, :], np.array(idx_targs_optimal), -1)
                idx_targs_optimal = [idx_targs_optimal[index] for index in (-discounts_targs_optimal).argsort().tolist()]

                if debug:
                    if self.waypoint_curr is None:
                        if self.cvae is None:
                            self.waypoint_curr = self.encoder_wp(obs_curr, env)
                        else:
                            self.waypoint_curr = self.encoder_wp(self.obs2tensor(obs_curr))
                    dist2targ = np.abs(self.waypoint_curr - self.waypoints_existing[idx_targ - 1]).sum()
                    writer.add_scalar(f"{prefix_plan}/dist2targ", dist2targ, step_record)
                    writer.add_scalar(
                        f"{prefix_plan}/dist2targ_robust",
                        np.abs(self.waypoint_curr - self.waypoints_existing[idx_targs_optimal[0] - 1]).sum(),
                        step_record,
                    )
                    writer.add_scalar(f"{prefix_plan}/deviation_Q_optimal", np.abs(Q_wp_curr_GT[idx_targs_optimal[0]] - Q_wp_curr[idx_targs[0]]), step_record)
                    writer.add_scalar(f"{prefix_plan}/deviation_Q_robust", np.abs(Q_wp_curr[idx_targs_optimal[0]] - Q_wp_curr[idx_targs[0]]), step_record)
                    if len(idx_targs_optimal):
                        plan_optimal = float(int(idx_targ) in idx_targs_optimal)
                        writer.add_scalar(f"{prefix_plan}/optimality", plan_optimal, step_record)
                        if len(idx_targs_optimal) > 1:
                            plan_optimal_robust = float(idx_targ == idx_targs_optimal[0])
                            writer.add_scalar(f"{prefix_plan}/optimality_robust", plan_optimal_robust, step_record)
                        mask_targs = np.zeros(self.num_waypoints, dtype=bool)
                        mask_targs[idx_targs] = True
                        mask_targs_optimal = np.zeros(self.num_waypoints, dtype=bool)
                        mask_targs_optimal[idx_targs_optimal] = True
                        writer.add_scalar(f"{prefix_plan}/optimal_intersect", (mask_targs == mask_targs_optimal).sum() / self.num_waypoints, step_record)

                    mask_interest = torch.logical_not(mask_cutoff)
                    mask_interest[:, 0] = False
                    mask_interest[omegas_GT] = False
                    mask_interest *= ~torch.eye(omegas_GT.shape[0], dtype=torch.bool, device=discounts_plan.device)
                    mask_existent_wps = np.ones(omegas_GT.shape[0], dtype=bool)
                    for idx_state in range(1, omegas_GT.shape[0]):
                        state = int(wp_graph_GT["states"][idx_state])
                        mask_existent_wps[idx_state] = state in env.DP_info["states_reachable"]
                    mask_existent_wps = torch.tensor(mask_existent_wps, device=discounts_plan.device)
                    mask_nonexistent_wps = torch.logical_not(mask_existent_wps)
                    mask_nonexistent = torch.zeros_like(mask_interest, dtype=torch.int64)
                    mask_nonexistent[mask_nonexistent_wps, :] += 1
                    mask_nonexistent[:, mask_nonexistent_wps] += 1
                    mask_nonexistent[mask_existent_wps, :] -= 1
                    mask_nonexistent[:, mask_existent_wps] -= 1
                    mask_nonexistent = mask_nonexistent == 0
                    mask_nonexistent[omegas_GT] = False

                    diff_distances = (
                        distances_GT.clamp(0, self.network_policy.estimator_discount.atoms)
                        - self.wp_graph_curr["distances"].clamp(0, self.network_policy.estimator_discount.atoms)
                    ).abs_()
                    diff_discounts = (discounts_GT - self.wp_graph_curr["discounts"]).abs_()
                    diff_rewards = (rewards_GT - self.wp_graph_curr["rewards"]).abs_()

                    if mask_interest.any():
                        deviation_Q = (Q_GT - Q).abs_()[mask_interest].mean().item()
                        writer.add_scalar(f"{prefix_plan}/deviation_Q", deviation_Q, step_record)
                        writer.add_scalar(f"{prefix_debug}/diff_distances", diff_distances[mask_interest].mean().item(), step_record)
                        writer.add_scalar(f"{prefix_debug}/diff_discounts", diff_discounts[mask_interest].mean().item(), step_record)
                        writer.add_scalar(f"{prefix_debug}/diff_rewards", diff_rewards[mask_interest].mean().item(), step_record)

                        mask_zero_discounts = discounts_GT == 0
                        mask_trivial_discounts = mask_zero_discounts * mask_interest
                        if mask_trivial_discounts.any():
                            diff_discounts_trivial = diff_discounts[mask_trivial_discounts]
                            writer.add_scalar(f"{prefix_debug}/diff_discounts_trivial", diff_discounts_trivial.mean().item(), step_record)

                        mask_nontrivial_discounts = ~mask_zero_discounts * mask_interest
                        if mask_nontrivial_discounts.any():
                            writer.add_scalar(f"{prefix_debug}/diff_discounts_nontrivial", diff_discounts[mask_nontrivial_discounts].mean().item(), step_record)

                        mask_zero_rewards = rewards_GT == 0
                        mask_trivial_rewards = mask_zero_rewards * mask_interest
                        if mask_trivial_rewards.any():
                            writer.add_scalar(f"{prefix_debug}/diff_rewards_trivial", diff_rewards[mask_trivial_rewards].mean().item(), step_record)
                        mask_nontrivial_rewards = ~mask_zero_rewards * mask_interest
                        if mask_nontrivial_rewards.any():
                            writer.add_scalar(f"{prefix_debug}/diff_rewards_nontrivial", diff_rewards[mask_nontrivial_rewards].mean().item(), step_record)

                    if mask_nonexistent.any():
                        writer.add_scalar(f"{prefix_debug}/diff_distances_nonexistent", diff_distances[mask_nonexistent].mean().item(), step_record)
                        writer.add_scalar(f"{prefix_debug}/diff_discounts_nonexistent", diff_discounts[mask_nonexistent].mean().item(), step_record)
                        writer.add_scalar(f"{prefix_debug}/diff_rewards_nonexistent", diff_rewards[mask_nonexistent].mean().item(), step_record)

                    writer.add_scalar(f"{prefix_debug}/diff_omegas", (omegas_GT != omegas_plan).float().mean().item(), step_record)
                if self.optimal_plan:
                    idx_targ = int(idx_targs_optimal[0])  # NOTE(H): try to pick the closest one
            assert idx_targ > 0, f"self-loop planned at step {self.steps_interact:d}: {self.waypoint_curr.tolist()}"
            if self.always_select_goal:
                idx_targ = len(self.waypoints_existing)
            self.idx_waypoint_targ = idx_targ - 1
            self.waypoint_targ = self.waypoints_existing[idx_targ - 1]
            self.wp_graph_curr["selected"][self.idx_waypoint_targ] = True
            self.obs_wp_targ = None if self.cvae is None else self.wp_existing_obses[[idx_targ - 1]]
            if self.optimal_policy:
                if self.pos_goal_oracle is None or (np.array(self.pos_goal_oracle) != np.array(self.waypoint_targ)).any():
                    ret = env.generate_oracle(goal_pos=self.wp_graph_curr["ijds"][idx_targ].tolist()[:2])
                    self.Q_oracle, self.pos_goal_oracle = ret["Q_optimal"], ret["goal_pos"]
            if debug:
                writer.add_scalar(f"{prefix_plan}/targ_valid", float(mask_existent_wps[idx_targ]), step_record)
        if self.optimal_policy:
            assert self.Q_oracle is not None and self.pos_goal_oracle is not None
            q = self.Q_oracle[env.obs2state(obs_curr)]
            if (q == 0).all():
                return self.action_space.sample()
            else:
                return q.argmax()
        if obs_curr_tensor is None:
            obs_curr_tensor = self.obs2tensor(obs_curr)
        if self.cvae is None:
            return self.Q_conditioned(obs_curr_tensor, waypoint_targ=self.waypoint_targ, obs_targ=None, type_curr="obs", env=env).argmax().item()
        else:
            return self.Q_conditioned(obs_curr_tensor, waypoint_targ=None, obs_targ=self.obs_wp_targ, type_curr="obs", env=env).argmax().item()

    def step(self, obs_curr, action, reward, obs_next, done, writer=None, add_to_buffer=True, increment_steps=True):
        if increment_steps:
            self.steps_interact += 1
        if add_to_buffer and obs_next is not None:
            sample = {"obs": np.array(obs_curr), "act": action, "rew": reward, "done": done, "next_obs": np.array(obs_next)}
            self.add_to_buffer(sample)


class SKIPPER(SKIPPER_BASE):
    def __init__(
        self,
        env,
        network_policy,
        network_target=None,
        freq_plan=4,
        num_waypoints=16,
        waypoint_strategy="once",
        always_select_goal=False,
        optimal_plan=False,
        optimal_policy=False,
        dist_cutoff=8,
        prune_with_oracle=False,
        gamma=0.99,
        gamma_int=0.95,
        type_intrinsic_reward="sparse",
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
        valid_waypoints_only=False,
        no_lava_waypoints=False,
        hrb=None,
        silent=False,
        transform_discount_target=True,
        num_waypoints_unpruned=32,
        suppress_delusion=False,
        no_Q_head=False,
        unique_codes=False,
        unique_obses=True,
    ):
        super(SKIPPER, self).__init__(
            env,
            network_policy,
            freq_plan=freq_plan,
            num_waypoints=num_waypoints,
            waypoint_strategy=waypoint_strategy,
            always_select_goal=always_select_goal,
            optimal_plan=optimal_plan,
            optimal_policy=optimal_policy,
            dist_cutoff=dist_cutoff,
            prune_with_oracle=prune_with_oracle,
            gamma=gamma,
            gamma_int=gamma_int,
            type_intrinsic_reward=type_intrinsic_reward,
            clip_reward=clip_reward,
            exploration_fraction=exploration_fraction,
            epsilon_final_train=epsilon_final_train,
            epsilon_eval=epsilon_eval,
            steps_total=steps_total,
            prioritized_replay=prioritized_replay,
            func_obs2tensor=func_obs2tensor,
            device=device,
            seed=seed,
            valid_waypoints_only=valid_waypoints_only,
            no_lava_waypoints=no_lava_waypoints,
            hrb=hrb,
            silent=silent,
            transform_discount_target=transform_discount_target,
            num_waypoints_unpruned=num_waypoints_unpruned,
            suppress_delusion=suppress_delusion,
            no_Q_head=no_Q_head,
            unique_codes=unique_codes,
            unique_obses=unique_obses,
        )

        self.optimizer = eval("torch.optim.%s" % type_optimizer)(self.network_policy.parameters(), lr=lr, eps=eps)
        # self.scheduler = torch.optim.lr_scheduler.LinearLR(
        #     self.optimizer, start_factor=1.0, end_factor=0.25, total_iters=exploration_fraction * steps_total / freq_train
        # )

        # initialize target network
        if network_target is None:
            self.network_target = copy.deepcopy(self.network_policy)
        else:
            self.network_target = network_target
        # self.network_target.to(self.device)
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
        (
            priorities,
            loss_TD,
            loss_discount,
            loss_reward,
            loss_omega,
            omega_logits_pred,
            batch_state_curr,
            state_local_curr,
        ) = self.calculate_multihead_error(batch_obs_curr, batch_action, batch_reward, batch_obs_next, batch_done, batch_obs_targ, batch_reward_int=None)
        if self.suppress_delusion:
            with torch.no_grad():
                if self.cvae is None:
                    _batch_obs_targ = torch.zeros_like(batch_obs_targ)
                    for i in range(self.size_batch):
                        env = copy.deepcopy(self.env)
                        obs_curr = batch_obs_curr[i].cpu().numpy()
                        env.load_layout_from_obs(obs_curr)
                        targ = generate_random_waypoints(
                            env,
                            1,
                            include_goal=False,
                            include_agent=False,
                            generate_DP_info=False,
                            render=False,
                            valid_only=False,
                            no_lava=True,
                            return_dist=False,
                            return_obs=True,
                            unique=False,
                            obs_curr=batch_obs_curr[i].cpu().numpy(),
                        )  # NOTE: using oracle
                        _batch_obs_targ[i] = self.obs2tensor(targ["obses"])
                        del env, obs_curr, targ
                else:
                    _batch_obs_targ = self.cvae.imagine_batch_from_obs(batch_obs_curr)
            (_, _loss_TD, _loss_discount, _loss_reward, _, _, _, _) = self.calculate_multihead_error(  # TODO(H): do two passes for now, check if it's good
                batch_obs_curr,
                batch_action,
                batch_reward,
                batch_obs_next,
                batch_done,
                _batch_obs_targ.detach(),
                batch_reward_int=None,
                calculate_Q_error=False,
                calculate_reward_error=False,
                calculate_omega_error=False,
                calculate_priorities=False,
                freeze_encoder=False,
                freeze_binder=False,
            )
            loss_TD_aux = _loss_TD.mean()
            loss_discount_aux = _loss_discount.mean()
            loss_reward_aux = _loss_reward.mean()

        if self.cvae is not None:
            (
                loss_cvae,
                loss_recon,
                loss_entropy,
                loss_conditional_prior,
                loss_align,
                dist_L1_mean,
                dist_L1_nontrivial,
                dist_L1_trivial,
                uniformity,
                entropy_prior,
                ratio_perfect_recon,
                ratio_aligned,
            ) = self.cvae.compute_loss(batch_processed, debug=debug)

        loss_overall = loss_TD + loss_discount + loss_reward + loss_omega
        if self.prioritized_replay:
            assert weights is not None
            # kaixhin's rainbow implementation used mean()
            error_overall_weighted = (loss_overall * weights.detach()).mean()

        else:
            error_overall_weighted = loss_overall.mean()

        if self.suppress_delusion:
            error_overall_weighted += 0.25 * loss_discount_aux
        if self.cvae is not None:
            error_overall_weighted += loss_cvae.mean()

        self.optimizer.zero_grad(set_to_none=True)
        error_overall_weighted.backward()

        # # gradient clipping
        # with torch.no_grad():
        #     if debug:
        #         norm_grad_sq = 0.0
        #     for param in self.network_policy.parameters():
        #         if debug:
        #             norm_grad_sq += (param.grad.data.detach() ** 2).sum().item()
        # norm_grad = np.sqrt(norm_grad_sq)
        # norm_grad = torch.nn.utils.clip_grad_norm_(self.network_policy.parameters(), 1.0)
        if debug:
            with torch.no_grad():
                grads = [param.grad.detach().flatten() for param in self.network_policy.parameters() if param.grad is not None]
                norm_grad = torch.cat(grads).norm().item()
        torch.nn.utils.clip_grad_value_(self.network_policy.parameters(), 1.0)
        self.optimizer.step()
        # self.scheduler.step()

        with torch.no_grad():
            # update prioritized replay, if used
            if self.prioritized_replay:
                # new_priorities = self.calculate_priorities(batch_obs_curr, batch_action, batch_reward, batch_obs_next, batch_done, batch_obs_targ, error_absTD=None)
                self.hrb.update_priorities(batch_idxes, priorities.detach().cpu().numpy())
                if debug:
                    writer.add_scalar("Train/priorities", priorities.mean().item(), self.steps_processed)

            if debug:
                if self.cvae is not None:
                    writer.add_scalar("Train_CVAE/loss_overall", loss_cvae.mean().item(), self.steps_processed)
                    writer.add_scalar("Train_CVAE/loss_recon", loss_recon.mean().item(), self.steps_processed)
                    writer.add_scalar("Train_CVAE/loss_entropy", loss_entropy.mean().item(), self.steps_processed)
                    if loss_align is not None:
                        writer.add_scalar("Train_CVAE/loss_align", loss_align.mean().item(), self.steps_processed)
                    writer.add_scalar("Train_CVAE/dist_L1", dist_L1_mean.item(), self.steps_processed)
                    writer.add_scalar("Train_CVAE/dist_L1_nontrivial", dist_L1_nontrivial.item(), self.steps_processed)
                    writer.add_scalar("Train_CVAE/dist_L1_trivial", dist_L1_trivial.item(), self.steps_processed)
                    writer.add_scalar("Train_CVAE/uniformity", uniformity.item(), self.steps_processed)
                    writer.add_scalar("Train_CVAE/ratio_imperfect_recon", 1.0 - ratio_perfect_recon.item(), self.steps_processed)
                    writer.add_scalar("Train_CVAE/ratio_aligned", ratio_aligned.item(), self.steps_processed)

                writer.add_scalar("Debug/norm_rep", torch.sqrt((batch_state_curr.flatten(1) ** 2).sum(-1)).mean().item(), self.steps_processed)
                writer.add_scalar("Debug/norm_rep_local", torch.sqrt((state_local_curr**2).sum(-1)).mean().item(), self.steps_processed)
                writer.add_scalar("Debug/norm_grad", norm_grad, self.steps_processed)
                writer.add_scalar("Train/loss_TD", loss_TD.mean().item(), self.steps_processed)
                writer.add_scalar("Train/loss_discount", loss_discount.mean().item(), self.steps_processed)
                writer.add_scalar("Train/loss_reward", loss_reward.mean().item(), self.steps_processed)
                writer.add_scalar("Train/loss_omega", loss_omega.mean().item(), self.steps_processed)
                # writer.add_scalar("Train/lr", self.scheduler.get_last_lr()[0], self.steps_processed)
                if omega_logits_pred is not None:
                    omega_pred = omega_logits_pred.argmax(-1).bool()
                    acc_omega = (batch_done == omega_pred).sum() / batch_done.shape[0]
                    writer.add_scalar("Debug/acc_omega", acc_omega.item(), self.steps_processed)

    def sync_parameters(self):
        """
        synchronize the parameters of self.network_policy and self.network_target
        this is hard sync, maybe a softer version is going to do better
        cvae not synced, since we don't need it for target network
        """
        self.network_target.encoder.load_state_dict(self.network_policy.encoder.state_dict())
        self.network_target.binder.load_state_dict(self.network_policy.binder.state_dict())
        if self.network_policy.estimator_Q is not None:
            self.network_target.estimator_Q.load_state_dict(self.network_policy.estimator_Q.state_dict())
        self.network_target.estimator_discount.load_state_dict(self.network_policy.estimator_discount.state_dict())
        self.network_target.estimator_reward.load_state_dict(self.network_policy.estimator_reward.state_dict())
        self.network_target.estimator_omega.load_state_dict(self.network_policy.estimator_omega.state_dict())
        if not self.silent:
            print("policy-target parameters synced")


def create_Skipper_network(args, env, dim_embed, num_actions, device, share_memory=False):
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
    if share_memory:
        encoder.share_memory()

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
    if share_memory:
        binder.share_memory()

    if args.no_Q_head:
        estimator_Q = None
    else:
        if args.type_intrinsic_reward == "sparse":
            dict_head_Q = {"len_predict": num_actions, "dist_out": True, "value_min": 0, "value_max": 1, "atoms": args.atoms_value, "classify": False}
        elif args.type_intrinsic_reward == "dense":
            dict_head_Q = {
                "len_predict": num_actions,
                "dist_out": True,
                "value_min": -float(args.atoms_value),
                "value_max": -1,
                "atoms": args.atoms_value,
                "classify": False,
            }
        else:
            raise NotImplementedError()

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
        if share_memory:
            estimator_Q.share_memory()

    if args.transform_discount_target:
        dict_head_discount = {
            "len_predict": num_actions,
            "dist_out": True,
            "value_min": 1,
            "value_max": args.atoms_discount,
            "atoms": args.atoms_discount,
            "classify": False,
        }
    else:
        dict_head_discount = {
            "len_predict": num_actions,
            "dist_out": True,
            "value_min": 0,
            "value_max": args.gamma,
            "atoms": args.atoms_discount,
            "classify": False,
        }
    estimator_discount = Predictor_MiniGrid(
        num_actions,
        len_input=binder.len_out,
        depth=args.depth_hidden,
        width=args.width_hidden,
        norm=bool(args.layernorm),
        activation=activation,
        dict_head=dict_head_discount,
    )
    if args.transform_discount_target:
        estimator_discount.histogram_converter.support_distance = torch.arange(1, args.atoms_discount + 1, device=device, dtype=torch.float32)
        estimator_discount.histogram_converter.support_discount = torch.pow(args.gamma, estimator_discount.histogram_converter.support_distance)
        # estimator_discount.histogram_converter.support_discount[-1] = 0.0
    else:
        estimator_discount.histogram_converter.support_discount = torch.linspace(0, args.gamma, args.atoms_discount, device=device, dtype=torch.float32)
        estimator_discount.histogram_converter.support_distance = torch.log(estimator_discount.histogram_converter.support_discount) / np.log(args.gamma)
        estimator_discount.histogram_converter.support_distance.clamp_(1, 250)
    estimator_discount.histogram_converter.support_override = True
    estimator_discount.to(device)
    if share_memory:
        estimator_discount.share_memory()

    dict_head_reward = {
        "len_predict": num_actions,
        "dist_out": True,
        "value_min": args.value_min,
        "value_max": args.value_max,
        "atoms": args.atoms_reward,
        "classify": False,
    }
    estimator_reward = Predictor_MiniGrid(
        num_actions,
        len_input=binder.len_out,
        depth=args.depth_hidden,
        width=args.width_hidden,
        norm=bool(args.layernorm),
        activation=activation,
        dict_head=dict_head_reward,
    )
    estimator_reward.to(device)
    if share_memory:
        estimator_reward.share_memory()

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

    if args.cvae:
        from models import Encoder_MiniGrid_Separate, Decoder_MiniGrid_Separate, CVAE_MiniGrid_Separate2

        encoder_CVAE = Encoder_MiniGrid_Separate()
        decoder_CVAE = Decoder_MiniGrid_Separate()

        num_categoricals, num_categories = 6, 2
        beta = 0.00025
        interval_beta = 10000

        cvae = CVAE_MiniGrid_Separate2(
            encoder_CVAE,
            decoder_CVAE,
            minigridobs2tensor(env.reset()),
            num_categoricals=num_categoricals,
            num_categories=num_categories,
            beta=beta,
            activation=activation,
            interval_beta=interval_beta,
        )
        cvae.to(device)
        if share_memory:
            cvae.share_memory()
    else:
        cvae = None

    network_policy = SKIPPER_NETWORK(encoder, binder, estimator_Q, estimator_discount, estimator_reward, estimator_omega, cvae=cvae)
    if share_memory:
        network_policy.share_memory()
    return network_policy


def create_Skipper_agent(
    args, env, dim_embed, num_actions, device=None, hrb=None, network_policy=None, network_target=None, inference_only=False, silent=False
):
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

    if network_policy is None:
        network_policy = create_Skipper_network(args, env, dim_embed, num_actions, device=device, share_memory=False)

    if inference_only:
        # TODO(H): maybe input all the CVAE hyperparameters here too
        agent = SKIPPER_BASE(
            env,
            network_policy,
            freq_plan=args.freq_plan,
            num_waypoints=args.num_waypoints,
            waypoint_strategy=args.waypoint_strategy,
            always_select_goal=args.always_select_goal,
            optimal_plan=args.optimal_plan,
            optimal_policy=args.optimal_policy,
            prune_with_oracle=args.prune_with_oracle,
            gamma=args.gamma,
            gamma_int=args.gamma_int,
            type_intrinsic_reward=args.type_intrinsic_reward,
            steps_total=args.steps_max,
            prioritized_replay=bool(args.prioritized_replay),
            device=device,
            seed=args.seed,
            valid_waypoints_only=args.valid_waypoints_only,
            no_lava_waypoints=args.no_lava_waypoints,
            hrb=hrb,
            silent=silent,
            transform_discount_target=args.transform_discount_target,
            num_waypoints_unpruned=args.num_waypoints_unpruned,
            suppress_delusion=args.suppress_delusion,
            no_Q_head=args.no_Q_head,
            unique_codes=args.unique_codes,
            unique_obses=args.unique_obses,
        )
    else:
        agent = SKIPPER(
            env,
            network_policy,
            freq_plan=args.freq_plan,
            num_waypoints=args.num_waypoints,
            waypoint_strategy=args.waypoint_strategy,
            always_select_goal=args.always_select_goal,
            optimal_plan=args.optimal_plan,
            optimal_policy=args.optimal_policy,
            prune_with_oracle=args.prune_with_oracle,
            gamma=args.gamma,
            gamma_int=args.gamma_int,
            type_intrinsic_reward=args.type_intrinsic_reward,
            steps_total=args.steps_max,
            prioritized_replay=bool(args.prioritized_replay),
            freq_train=args.freq_train,
            freq_targetsync=args.freq_targetsync,
            lr=args.lr,
            size_batch=args.size_batch,
            device=device,
            seed=args.seed,
            valid_waypoints_only=args.valid_waypoints_only,
            no_lava_waypoints=args.no_lava_waypoints,
            hrb=hrb,
            silent=silent,
            transform_discount_target=args.transform_discount_target,
            num_waypoints_unpruned=args.num_waypoints_unpruned,
            suppress_delusion=args.suppress_delusion,
            no_Q_head=args.no_Q_head,
            unique_codes=args.unique_codes,
            unique_obses=args.unique_obses,
        )
    return agent
