import gym, torch, numpy as np, copy, time
from minigrid import OBJECT_TO_IDX  # , COLOR_TO_IDX
from visual_utils import outline, gen_comparative_image
import matplotlib.pyplot as plt

# from collections import deque
from cpprb import PrioritizedReplayBuffer, ReplayBuffer
from HER import HindsightReplayBuffer

import queue

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def check_reachability_conditioned(env, agent, state_start, state_targ, batch_state_all):
    obs_targ = env.state2obs(state_targ)
    q_s_a = agent.Q_conditioned(
        batch_state_all,
        obs_targ=obs_targ,
        type_curr="state_rep",
    )  # assuming that policy is a |S| x |A| matrix
    policy = torch.nn.functional.one_hot(q_s_a.argmax(1), env.DP_info["P"].shape[0]).float()
    P_pi_targ = torch.einsum(
        "ijk,ji->jk",
        torch.tensor(env.DP_info["P"][:, env.DP_info["states_reachable"], :], device=agent.device, dtype=policy.dtype),
        policy,
    )
    P_pi_targ = P_pi_targ[:, env.DP_info["states_reachable"]]
    idx_state_start = env.DP_info["states_reachable"].index(state_start)
    idx_state_targ = env.DP_info["states_reachable"].index(state_targ)
    steps_expected = reachability_BFS(P_pi_targ.bool().cpu().numpy(), idx_state_start, idx_state_targ)
    return steps_expected


@torch.no_grad()
# @profile
def evaluate_multihead_minigrid(
    env,
    agent,
    writer,
    size_batch=32,
    num_episodes=10,
    suffix="",
    step_record=None,
    env_generator=None,
    max_dist=16,
    queue_envs=None,
):
    time_start = time.time()
    if step_record is None:
        step_record = agent.steps_interact
    (
        dict_suboptimality_vs_dist,
        dict_error_discount_vs_dist,
        dict_error_reward_vs_dist,
        dict_failure_vs_dist,
        dict_error_distance_vs_dist,
    ) = ({}, {}, {}, {}, {})
    for dist in range(max_dist):
        dict_suboptimality_vs_dist[f"{dist}"] = []
        dict_failure_vs_dist[f"{dist}"] = []
        dict_error_discount_vs_dist[f"{dist}"] = []
        dict_error_distance_vs_dist[f"{dist}"] = []
        dict_error_reward_vs_dist[f"{dist}"] = []
    for episode in range(num_episodes):
        if queue_envs is not None:
            env = None
            while env is None:
                try:
                    env = queue_envs.get_nowait()
                except:
                    pass
        elif env_generator is not None:
            env = env_generator()
            env.reset()
        else:
            env.reset()
        if env.DP_info["state_target_tuples"] is None:
            env.generate_state_target_tuples(max_dist=max_dist)  # to generate the (s, s_targ, dist) tuples
        assert env.DP_info["state_target_tuples"] is not None

        list_tuples = env.DP_info["state_target_tuples"]
        size_batch_actual = min(size_batch, len(list_tuples))
        indices_sampled = np.random.choice(len(list_tuples), size_batch_actual)
        tuples_sampled = []
        for index in indices_sampled:
            tuples_sampled.append(list_tuples[index])
        # constuct batches
        states_curr = [tuple[0] for tuple in tuples_sampled]
        states_targ = [tuple[1] for tuple in tuples_sampled]
        batch_obs_curr = env.state2obs(states_curr)
        batch_obs_targ, batch_ijd_targ = env.state2obs(states_targ, return_ijd=True)
        batch_obs_curr_targ = torch.cat([agent.obs2tensor(batch_obs_curr), agent.obs2tensor(batch_obs_targ)], 0)
        batch_state_curr_targ = agent.network_policy.encoder(batch_obs_curr_targ)
        batch_state_curr, batch_state_targ = torch.split(batch_state_curr_targ, [size_batch_actual, size_batch_actual], dim=0)
        tuples_curr_targ, state_local_curr = agent.network_policy.binder(batch_state_curr, batch_state_targ, return_curr=True)

        predicted = {}
        if agent.network_policy.estimator_Q is None:
            predicted_discounts = agent.network_policy.estimator_discount(tuples_curr_targ, scalarize=False).softmax(-1) @ agent.support_discount
            actions = predicted_discounts.argmax(-1, keepdim=True)
        else:
            predicted["Q"] = agent.network_policy.estimator_Q(tuples_curr_targ, scalarize=True)
            actions = predicted["Q"].argmax(-1, keepdim=True)

        predicted_target_discount_dist = agent.network_target.estimator_discount(tuples_curr_targ, actions, scalarize=False)
        softmax_target_discount_dist = predicted_target_discount_dist.softmax(-1)
        predicted["distance"] = softmax_target_discount_dist @ agent.network_target.estimator_discount.histogram_converter.support_distance
        predicted["discount"] = softmax_target_discount_dist @ agent.network_target.estimator_discount.histogram_converter.support_discount

        predicted["reward"] = agent.network_policy.estimator_reward(tuples_curr_targ, actions, scalarize=True)

        predicted["omega"] = agent.network_policy.estimator_omega(state_local_curr, scalarize=True).bool().squeeze()
        predicted["discount"][predicted["omega"]] = 0.0
        predicted["reward"][predicted["omega"]] = 0.0
        predicted["distance"][predicted["omega"]] = max_dist - 1

        if env.DP_info["P"] is None:
            env.collect_transition_probs()
        if env.DP_info["obses_all"] is None:
            # NOTE(H): this actually makes sense, since the number of states are not too much
            env.generate_obses_all()

        batch_obs_all = agent.obs2tensor(env.DP_info["obses_all"])
        batch_state_all = agent.network_policy.encoder(batch_obs_all)

        ijds_targ_sampled = np.stack(batch_ijd_targ, 1)
        for idx_tuple in range(len(tuples_sampled)):
            tuple = tuples_sampled[idx_tuple]
            state_curr, state_targ, dist = tuple
            ijd_targ = ijds_targ_sampled[idx_tuple]  # env.state2ijd(state_targ)
            DP_info = env.generate_oracle(goal_pos=(int(ijd_targ[0]), int(ijd_targ[1])))
            suboptimality = 1.0 - env.evaluate_action(int(actions[idx_tuple]), obs=batch_obs_curr[idx_tuple], DP_info=DP_info)

            steps_expected = check_reachability_conditioned(env, agent, state_curr, state_targ, batch_state_all)
            if np.isinf(steps_expected):  # TODO(H): find better comparison metrics
                failure = 1.0
            else:
                failure = 0.0
            discount_true = env.gamma**dist
            error_discount = np.abs(discount_true - predicted["discount"][idx_tuple].detach().cpu().numpy())
            dict_error_discount_vs_dist[f"{dist}"].append(float(error_discount))

            error_distance = np.abs(min(dist, max_dist - 1) - predicted["distance"][idx_tuple].detach().cpu().numpy())
            dict_error_distance_vs_dist[f"{dist}"].append(float(error_distance))
            i_targ, j_targ, _ = ijd_targ
            i_goal, j_goal = env.goal_pos
            if i_targ == i_goal and j_targ == j_goal:
                G_real = float(np.max(DP_info["Q_optimal"][state_curr]))
            else:
                G_real = 0.0
            error_reward = np.abs(G_real - predicted["reward"][idx_tuple].detach().cpu().numpy())
            dict_suboptimality_vs_dist[f"{dist}"].append(float(suboptimality))
            dict_failure_vs_dist[f"{dist}"].append(float(failure))
            dict_error_reward_vs_dist[f"{dist}"].append(float(error_reward))

    vec_suboptimality_vs_dist = np.full(max_dist, np.nan)
    for i, (k, v) in enumerate(dict_suboptimality_vs_dist.items()):
        if not len(v):
            continue
        v = np.array(v)
        elements_not_nan = np.take(v, np.where(np.logical_not(np.isnan(v))))
        if len(elements_not_nan):
            vec_suboptimality_vs_dist[int(k)] = np.mean(elements_not_nan)

    vec_failure_vs_dist = np.full(max_dist, np.nan)
    for i, (k, v) in enumerate(dict_failure_vs_dist.items()):
        if not len(v):
            continue
        v = np.array(v)
        elements_not_nan = np.take(v, np.where(np.logical_not(np.isnan(v))))
        if len(elements_not_nan):
            vec_failure_vs_dist[int(k)] = np.mean(elements_not_nan)

    vec_error_discount_vs_dist = np.zeros(max_dist)
    for i, (k, v) in enumerate(dict_error_discount_vs_dist.items()):
        if not len(v):
            continue
        v = np.array(v)
        elements_not_nan = np.take(v, np.where(np.logical_not(np.isnan(v))))
        if len(elements_not_nan):
            vec_error_discount_vs_dist[int(k)] = np.mean(elements_not_nan)

    vec_error_distance_vs_dist = np.zeros(max_dist)
    for i, (k, v) in enumerate(dict_error_distance_vs_dist.items()):
        if not len(v):
            continue
        v = np.array(v)
        elements_not_nan = np.take(v, np.where(np.logical_not(np.isnan(v))))
        if len(elements_not_nan):
            vec_error_distance_vs_dist[int(k)] = np.mean(elements_not_nan)

    vec_error_reward_vs_dist = np.zeros(max_dist)
    for i, (k, v) in enumerate(dict_error_reward_vs_dist.items()):
        if not len(v):
            continue
        v = np.array(v)
        elements_not_nan = np.take(v, np.where(np.logical_not(np.isnan(v))))
        if len(elements_not_nan):
            vec_error_reward_vs_dist[int(k)] = np.mean(elements_not_nan)

    fig, ax = plt.subplots()
    ax.plot(np.arange(max_dist), vec_suboptimality_vs_dist)
    ax.set_xlim(0, max_dist - 1)
    ax.set_ylim(0, 1)
    writer.add_figure("DP/suboptimality" + suffix, fig, step_record)
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(np.arange(max_dist), vec_failure_vs_dist)
    ax.set_xlim(0, max_dist - 1)
    ax.set_ylim(0, 1)
    writer.add_figure("DP/failure" + suffix, fig, step_record)
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(np.arange(max_dist), vec_error_discount_vs_dist)
    ax.set_xlim(0, max_dist - 1)
    ax.set_ylim(1e-4, 1)
    ax.set_yscale("log")
    writer.add_figure("DP/error_discount" + suffix, fig, step_record)
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(np.arange(max_dist), vec_error_distance_vs_dist)
    ax.set_xlim(0, max_dist - 1)
    ax.set_ylim(0, max_dist - 1)
    writer.add_figure("DP/error_distance" + suffix, fig, step_record)
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(np.arange(max_dist), vec_error_reward_vs_dist)
    ax.set_xlim(0, max_dist - 1)
    ax.set_ylim(1e-4, 1)
    ax.set_yscale("log")
    writer.add_figure("DP/error_reward" + suffix, fig, step_record)
    plt.close(fig)

    time_elapsed = time.time() - time_start
    print(f"(evaluate_multihead_minigrid): {num_episodes:d}x{size_batch:d} done in {time_elapsed:.2g}s")
    return {
        "suboptimality": vec_suboptimality_vs_dist,
        "failure": vec_failure_vs_dist,
        "discount": vec_error_discount_vs_dist,
        "distance": vec_error_distance_vs_dist,
        "reward": vec_error_reward_vs_dist,
    }


@torch.no_grad()
def minigridobs2tensor(obs, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    tensor = torch.tensor(obs, device=device)
    if len(tensor.shape) == 3:
        tensor = torch.unsqueeze(tensor, 0)
    assert len(tensor.shape) == 4
    return tensor


def get_cpprb(env, size_buffer, prioritized=False, hindsight=False, hindsight_strategy="future", ctx=None):
    env_dict = get_cpprb_env_dict(env)
    if hindsight:
        return HindsightReplayBuffer(
            4 * size_buffer,
            env_dict,
            max_episode_len=env.unwrapped.max_steps,
            reward_func=None,
            prioritized=prioritized,
            strategy=hindsight_strategy,
            ctx=ctx,
        )
    else:
        if prioritized:
            return PrioritizedReplayBuffer(size_buffer, env_dict, ctx=ctx)
        else:
            return ReplayBuffer(size_buffer, env_dict, ctx=ctx)


def get_space_size(space):
    if isinstance(space, gym.spaces.box.Box):
        return space.shape
    elif isinstance(space, gym.spaces.discrete.Discrete):
        return [1]
    else:
        raise NotImplementedError("Assuming to use Box or Discrete, not {}".format(type(space)))


def get_default_rb_dict(size, env):
    return {
        "size": size,
        "default_dtype": np.float32,
        "env_dict": {
            "obs": {"shape": get_space_size(env.observation_space)},
            "next_obs": {"shape": get_space_size(env.observation_space)},
            "act": {"shape": get_space_size(env.action_space)},
            "rew": {},
            "done": {},
        },
    }


def get_cpprb_env_dict(env):
    shape_obs = get_space_size(env.observation_space)
    env_dict = {
        "obs": {"shape": shape_obs},
        "act": {},
        "rew": {"shape": 1},
        "done": {"shape": 1, "dtype": bool},
    }
    if isinstance(env.action_space, gym.spaces.discrete.Discrete):
        env_dict["act"]["shape"] = 1
        env_dict["act"]["dtype"] = np.uint8
    elif isinstance(env.action_space, gym.spaces.box.Box):
        env_dict["act"]["shape"] = env.action_space.shape
        env_dict["act"]["dtype"] = np.float32
    obs = env.reset()
    if isinstance(obs, np.ndarray):
        env_dict["obs"]["dtype"] = obs.dtype
    # elif isinstance(obs, LazyFrames):
    #     env_dict["obs"]["dtype"] = obs._frames[0].dtype
    env_dict["next_obs"] = env_dict["obs"]
    return env_dict


class LinearSchedule(object):
    def __init__(self, schedule_timesteps, final_p, initial_p=1.0):
        self.schedule_timesteps = schedule_timesteps
        self.final_p = final_p
        self.initial_p = initial_p

    def value(self, t):
        fraction = max(0.0, min(float(t) / self.schedule_timesteps, 1.0))
        return self.initial_p + fraction * (self.final_p - self.initial_p)


@torch.no_grad()
def init_weights(architecture):
    for layer in architecture:
        if type(layer) == torch.nn.Linear:
            torch.nn.init.xavier_normal_(layer.weight)
            torch.nn.init.uniform_(layer.bias, -np.sqrt(1.0 / layer.in_features), np.sqrt(1.0 / layer.in_features))
        elif type(layer) == torch.nn.Conv1d:
            torch.nn.init.xavier_normal_(layer.weight)
            torch.nn.init.uniform_(layer.bias, -np.sqrt(1.0 / layer.in_channels), np.sqrt(1.0 / layer.in_channels))


def dijkstra(graph, start, max_dist=float("inf")):
    """
    Implementation of dijkstra using adjacency matrix.
    This returns an array containing the length of the shortest path from the start node to each other node.
    It is only guaranteed to return correct results if there are no negative edges in the graph. Positive cycles are fine.
    This has a runtime of O(|V|^2) (|V| = number of Nodes), for a faster implementation see @see ../fast/Dijkstra.java (using adjacency lists)

    :param graph: an adjacency-matrix-representation of the graph where (x,y) is the weight of the edge or 0 if there is no edge.
    :param start: the node to start from.
    :return: an array containing the shortest distances from the given start node to each other node
    """
    # This contains the distances from the start node to all other nodes
    distances = [max_dist for _ in range(len(graph))]

    # This contains whether a node was already visited
    visited = [False for _ in range(len(graph))]

    # The distance from the start node to itself is of course 0
    distances[start] = 0

    # While there are nodes left to visit...
    while True:
        # ... find the node with the currently shortest distance from the start node...
        shortest_distance = max_dist
        shortest_index = -1
        for i in range(len(graph)):
            # ... by going through all nodes that haven't been visited yet
            if distances[i] < shortest_distance and not visited[i]:
                shortest_distance = distances[i]
                shortest_index = i

        # print("Visiting node " + str(shortest_index) + " with current distance " + str(shortest_distance))

        if shortest_index == -1:
            # There was no node not yet visited --> We are done
            return distances

        # ...then, for all neighboring nodes that haven't been visited yet....
        for i in range(len(graph[shortest_index])):
            # ...if the path over this edge is shorter...
            if graph[shortest_index][i] != 0 and distances[i] > distances[shortest_index] + graph[shortest_index][i]:
                # ...Save this path as new shortest path.
                distances[i] = distances[shortest_index] + graph[shortest_index][i]
                # print("Updating distance of node " + str(i) + " to " + str(distances[i]))

        # Lastly, note that we are finished with this node.
        visited[shortest_index] = True
        # print("Visited nodes: " + str(visited))
        # print("Currently lowest distances: " + str(distances))


class HistogramConverter(torch.nn.Module):
    """
    consistent scalar <-> histogram converter for distributional outputs
    """

    def __init__(self, value_min=-1, value_max=1, atoms=128):
        super(HistogramConverter, self).__init__()
        self.register_buffer("value_min", torch.tensor(value_min))
        self.register_buffer("value_max", torch.tensor(value_max))
        self.atoms = atoms
        self.value_span = value_max - value_min
        const_norm = torch.tensor((self.atoms - 1) / self.value_span)
        self.register_buffer("const_norm", const_norm)
        const_norm_inv = torch.tensor(self.value_span / (self.atoms - 1))
        self.register_buffer("const_norm_inv", const_norm_inv)
        support = torch.arange(self.atoms).float()
        self.register_buffer("support", support)
        self.support_override = False

    def to(self, device):
        super().to(device)
        self.value_min = self.value_min.to(device)
        self.value_max = self.value_max.to(device)
        self.const_norm = self.const_norm.to(device)
        self.const_norm_inv = self.const_norm_inv.to(device)
        self.support = self.support.to(device)

    def parameters(self):
        return []

    @torch.no_grad()
    def to_histogram(self, value):
        value_normalized = (value - self.value_min) * self.const_norm  # normalize to [0, atoms - 1] range
        value_normalized.clamp_(0, self.atoms - 1)
        upper, lower = value_normalized.ceil().long(), value_normalized.floor().long()
        upper_weight = value_normalized % 1
        lower_weight = 1 - upper_weight
        dist = torch.zeros(value.shape[0], self.atoms, device=value.device, dtype=value.dtype)
        dist.scatter_add_(-1, lower, lower_weight)
        dist.scatter_add_(-1, upper, upper_weight)
        return dist  # validated with "self.from_histogram(dist, logits=False) - value.squeeze()"

    @torch.no_grad()
    def from_histogram(self, dist, logits=True):
        if logits:
            dist = torch.nn.functional.softmax(dist, -1)
        if self.support_override:
            value = dist @ self.support
            return value
        else:
            value_normalized = dist @ self.support
            value = self.value_min + value_normalized * self.const_norm_inv
            return value


class RL_AGENT(torch.nn.Module):
    def __init__(self, env, gamma, seed):
        super(RL_AGENT, self).__init__()
        self.gamma = gamma
        self.seed = seed
        self.observation_space, self.action_space = copy.deepcopy(env.observation_space), copy.deepcopy(env.action_space)
        self.env = env

    def on_episode_end(self, eval=False):
        pass


def distance_states(A, states, done):
    num_waypoints = len(states)
    dist = np.full([num_waypoints, num_waypoints], np.inf, dtype=np.float32)
    for idx_target in range(num_waypoints):
        ret = dijkstra(np.transpose(A), states[idx_target])
        dist[:, idx_target] = np.array(ret)[states]
    mask_all_invalid = np.logical_or(np.isinf(dist), dist == 0).all(-1)

    for idx_waypoint in range(num_waypoints):
        if not done[idx_waypoint] and not mask_all_invalid[idx_waypoint]:
            # NOTE(H): not terminal and can reach other states (and back)
            for idx_target in range(num_waypoints):
                if dist[idx_waypoint, idx_target] == 0:
                    # NOTE(H): djisktra returns 0 for self-loop
                    dist[idx_waypoint, idx_target] = 2 - int(A[idx_waypoint, idx_waypoint])
        else:
            dist[idx_waypoint, :] = np.inf
    return dist


def append_GT_graph(env, aux):
    aux = copy.deepcopy(aux)
    num_waypoints = aux["ijds"].shape[0]
    if isinstance(aux["states"], np.ndarray):
        list_states = aux["states"].tolist()
    elif isinstance(aux["states"], list):
        list_states = aux["states"]
    else:
        raise RuntimeError("Unknown type of states")
    ijds = aux["ijds"]
    if env.DP_info["states_reachable"] is None:
        env.collect_states_reachable()
    mask_valid_wps = np.ones(len(list_states), dtype=bool)
    for idx_state in range(len(list_states)):
        state = int(list_states[idx_state])
        mask_valid_wps[idx_state] = state in env.DP_info["states_reachable"]

    if env.DP_info["A"] is None:
        env.collect_state_adjacency()

    if "done" not in aux.keys() or aux["done"] is None:
        E = np.eye(env.DP_info["A"].shape[0]).astype(bool)
        done = np.all(env.DP_info["A"][list_states] == E[list_states], axis=-1)
        aux["done"] = done

    dist = distance_states(env.DP_info["A"], list_states, done)
    mask_valid2invalid = np.zeros_like(dist, dtype=bool)
    mask_valid2invalid[mask_valid_wps, :] = True
    mask_valid2invalid[:, mask_valid_wps] = False
    dist[mask_valid2invalid] = np.inf

    discount = env.gamma**dist  # NOTE(H): be careful of the self-loop, this is only a temporary solution

    reward_terminal = np.zeros(num_waypoints, dtype=np.float32)
    for idx_waypoint in range(num_waypoints):
        ijd = ijds[idx_waypoint]
        if ijd[0] == env.goal_pos[0] and ijd[1] == env.goal_pos[1]:
            reward_terminal[idx_waypoint] = 1.0

    # NOTE(H): this is not perfect but this is good for now
    reward = reward_terminal.reshape(1, -1) * env.gamma ** (dist - 1).clip(min=0)
    aux["distance"] = dist
    aux["discount"] = discount
    aux["reward"] = reward
    return aux


def floyd_warshall(A):
    dist = A.astype(np.float32)
    dist[dist == 0] = np.inf
    nV = dist.shape[0]
    dist[np.eye(nV, dtype=bool)] = 0.0
    for k in range(nV):
        dist = np.minimum(dist, dist[None, k, :] + dist[:, k, None])
    return dist


def generate_random_waypoints(
    env,
    num_waypoints,
    valid_only=False,
    include_agent=True,
    include_goal=True,
    generate_DP_info=True,
    render=True,
    no_lava=False,
    return_dist=False,
    return_obs=False,
    unique=False,
    obs_curr=None,
):
    env = env.unwrapped
    assert "RandDistShift" in env.spec.id
    aux = {}
    # with CVAE you will be able to generate code and obs pairs, with obs, we can always find the corresponding ijd
    # but if the goal is included and the goal_obs does not correspond to anything in the latent space
    # how should we deal with it?
    # luckily, we don't have to deal with this because we don't really have to check if we arrived at the goal (task terminates when that happens)
    # we should not translate wp to obs everytime, should keep the obs and translate obs back to the new abstract code
    # right now, return obs is False by default, since we don't know if later we would use prune by oracle or not, the obses are generated in agent.get_abstract_graph
    if valid_only or include_goal:
        if env.DP_info["states_reachable"] is None:
            env.collect_states_reachable()
        states_reachable = env.DP_info["states_reachable"]

    if include_goal:
        goal_i, goal_j = env.goal_pos
        i_agent, j_agent, d_agent = *env.agent_pos, env.agent_dir
        agent_state = env.ijd2state(int(i_agent), int(j_agent), d_agent)

        if env.ignore_dir:
            goal_states = [env.ijd2state(goal_i, goal_j, 0)]
        else:
            goal_states = np.unique([env.ijd2state(goal_i, goal_j, d) for d in range(4)])
        reachable_goal_states = np.intersect1d(goal_states, states_reachable).tolist()
        assert len(reachable_goal_states)
        if len(reachable_goal_states) == 1:
            nearest_goal_state = reachable_goal_states[0]
        else:
            dists_reachable_goal_states = []
            ret = dijkstra(env.DP_info["A"], agent_state)
            for state in reachable_goal_states:
                dists_reachable_goal_states.append(ret[state])
            idx_nearest_goal_state = np.argmin(dists_reachable_goal_states)
            nearest_goal_state = reachable_goal_states[idx_nearest_goal_state]
        nearest_goal_ijd = env.state2ijd(nearest_goal_state)
        if env.ignore_dir:
            nearest_goal_ijd = nearest_goal_ijd[:2]
    if include_agent:
        if env.ignore_dir:
            list_ijds = [np.array([*env.agent_pos]).astype(np.int32)]
        else:
            list_ijds = [np.array([*env.agent_pos, env.agent_dir]).astype(np.int32)]
    else:
        list_ijds = []

    while len(list_ijds) < num_waypoints:
        if include_goal and len(list_ijds) == num_waypoints - 1:  # last one to be the goal
            list_ijds.append(nearest_goal_ijd)
            break
        if env.ignore_dir:
            ijd = np.floor(np.random.rand(2) * np.array([env.width, env.height])).astype(np.int32)
        else:
            ijd = np.floor(np.random.rand(3) * np.array([env.width, env.height, 4])).astype(np.int32)
        can_add = True
        if valid_only:
            if env.ignore_dir and not env.ijd2state(int(ijd[0]), int(ijd[1])) in states_reachable:
                can_add = False
                continue
            elif not env.ignore_dir and not env.ijd2state(int(ijd[0]), int(ijd[1]), int(ijd[2])) in states_reachable:
                can_add = False
                continue
        if no_lava and env.DP_info["lava_map"][int(ijd[0]), int(ijd[1])]:
            can_add = False
            continue
        if unique:
            for ijd_prev in list_ijds:  # NOTE(H): guarantee uniqueness
                if ijd[0] == ijd_prev[0] and ijd[1] == ijd_prev[1] or ijd[0] == nearest_goal_ijd[0] and ijd[1] == nearest_goal_ijd[1]:
                    if "RandDistShift-v2" in env.spec.id or ijd[2] == ijd_prev[2] or ijd[2] == nearest_goal_ijd[2]:
                        can_add = False
                        break
        if can_add:
            list_ijds.append(ijd)
    ijds = np.stack(list_ijds, axis=0)
    # aux["states_reachable"] = states_reachable
    aux["ijds"] = ijds
    aux["codes"] = ijds

    list_states = []
    for idx_waypoint in range(num_waypoints):
        ijd = ijds[idx_waypoint]
        list_states.append(env.ijd2state(*ijd.tolist()))
    states = np.stack(list_states, axis=0)
    aux["states"] = states

    if return_obs:
        list_obses = []
        for state in list_states:
            list_obses.append(env.state2obs(state))
        obses = np.stack(list_obses, axis=0)
        if include_agent:
            assert (list_obses[0] == obs_curr).all()
    else:
        obses = None
    aux["obses"] = obses

    if render:
        highlight_mask = np.zeros([env.width, env.height], dtype=bool)
        for idx_waypoint in range(num_waypoints):
            ijd = ijds[idx_waypoint]
            highlight_mask[ijd[0], ijd[1]] = True
        rendered = env.grid.render(
            32,
            env.agent_pos,
            env.agent_dir,
            highlight_mask=highlight_mask,
            obs=env.obs_curr,
        )
        aux["rendered"] = np.flip(rendered, axis=0)
    else:
        aux["rendered"] = None

    if return_dist and not generate_DP_info:
        if "done" not in aux.keys() or aux["done"] is None:
            E = np.eye(env.DP_info["A"].shape[0]).astype(bool)
            done = np.all(env.DP_info["A"][list_states] == E[list_states], axis=-1)
            aux["done"] = done
        dist = distance_states(env.DP_info["A"], list_states, aux["done"])
        aux["distance"] = dist

    if generate_DP_info:
        aux = append_GT_graph(env, aux)

    return aux


@torch.no_grad()
def abstract_planning(gammas, rewards, omegas=None, tol=1e-5, max_iters=5, no_loop=True):
    """
    gammas: discount |S|x|S|
    rewards: option transition cumulative rewards |S|x|S|
    omegas: terminating probs |S|
    """

    num_abs_states = gammas.shape[0]
    gammas = gammas.squeeze()
    rewards = rewards.squeeze()

    omega_masking = omegas is not None and omegas.any()
    if omega_masking:
        gammas = gammas.clone()
        rewards = rewards.clone()
        if omega_masking:
            gammas[omegas] = 0
            rewards[omegas] = 0

    v_old = torch.zeros(num_abs_states, device=gammas.device, dtype=gammas.dtype)
    num_iters = 0
    while True:
        num_iters += 1
        q = rewards + gammas * v_old.reshape(1, -1)
        if no_loop:
            q.fill_diagonal_(0)  # TODO(H):change here if with negative rewards
        v_new = q.max(dim=-1)[0]
        converged = torch.allclose(v_new, v_old, rtol=tol, atol=tol, equal_nan=False)
        if num_iters >= max_iters or converged:
            break
        v_old = v_new
    return q, num_iters, converged


@torch.no_grad()
def process_batch(batch, prioritized=False, with_targ=False, device=DEVICE, obs2tensor=minigridobs2tensor, clip_reward=False, aux=False):
    # even with prioritized replay, one would still want to process a batch without the priorities
    batch_obs_curr = batch["obs"]
    batch_action = batch["act"]
    batch_reward = batch["rew"]
    batch_done = batch["done"]
    batch_obs_next = batch["next_obs"]
    if prioritized:
        weights = batch["weights"]
        batch_idxes = batch["indexes"]
        weights = torch.tensor(weights, dtype=torch.float32, device=device).reshape(-1, 1)
    else:
        weights, batch_idxes = None, None

    if aux:
        info = {}

    if with_targ:
        batch_obs_targ = batch["goal"]
        if aux:
            info = {}
            pointer = 1
            while True:
                if f"goal{pointer}" in batch.keys():
                    info[f"batch_obs_targ{pointer}"] = batch[f"goal{pointer}"]
                    pointer += 1
                else:
                    break

    batch_reward = torch.tensor(batch_reward, dtype=torch.float32, device=device).reshape(-1, 1)
    batch_done = torch.tensor(batch_done, dtype=torch.bool, device=device).reshape(-1)
    batch_action = torch.tensor(batch_action, dtype=torch.int64, device=device).reshape(-1, 1)

    batch_obs_curr, batch_obs_next = obs2tensor(batch_obs_curr, device=device), obs2tensor(batch_obs_next, device=device)
    if with_targ:
        batch_obs_targ = obs2tensor(batch_obs_targ, device=device)
        if aux:
            if len(info):
                for key, value in info.items():
                    info[key] = obs2tensor(value)

    if clip_reward:  # this is a DQN-specific thing
        batch_reward = torch.sign(batch_reward)

    if with_targ:
        ret = (batch_obs_curr, batch_action, batch_reward, batch_obs_next, batch_done, batch_obs_targ, weights, batch_idxes)
    else:
        ret = (batch_obs_curr, batch_action, batch_reward, batch_obs_next, batch_done, weights, batch_idxes)

    if aux:
        return *ret, info
    else:
        return ret


gray_encoding = torch.tensor(
    [
        [0, 0, 0, 0],  # 0
        [0, 0, 0, 1],  # 1
        [0, 0, 1, 1],  # 2
        [0, 0, 1, 0],  # 3
        [0, 1, 1, 0],  # 4
        [0, 1, 1, 1],  # 5
        [0, 1, 0, 1],  # 6
        [0, 1, 0, 0],  # 7
        [1, 1, 0, 0],  # 8
        [1, 1, 0, 1],  # 9
        [1, 1, 1, 1],  # 10
        [1, 1, 1, 0],  # 11
        [1, 0, 1, 0],  # 12
        [1, 0, 1, 1],  # 13
        [1, 0, 0, 1],  # 14
        [1, 0, 0, 0],  # 15
    ]
)


@torch.no_grad()
def debug_cvae_generation(obs_sampled_compact, obs_cond, env, writer, step_record, label="Possible Samples"):
    time_then = time.time()
    size_batch = obs_sampled_compact.shape[0]
    obs_sampled_compact = torch.round(obs_sampled_compact).detach().to(torch.uint8)
    slice_obs_sampled_compact = obs_sampled_compact[:, :, :, 0]
    slice_obs_cond = obs_cond[:, :, 0]
    mask_sampled_agent = (slice_obs_sampled_compact == OBJECT_TO_IDX["agent"]).detach().cpu().numpy()

    # mask_sampled_goal = (slice_obs_sampled_compact == OBJECT_TO_IDX["goal"]).detach().cpu().numpy()
    # num_agents_per_sample = mask_sampled_agent.reshape(size_batch, -1).sum(-1)

    mask_lava_reached = slice_obs_cond == OBJECT_TO_IDX["lava"]
    mask_goal_reached = slice_obs_cond == OBJECT_TO_IDX["goal"]

    mask_lava_reached_repeated = np.repeat(mask_lava_reached[None, :], size_batch, 0)
    mask_samples_lava_reached = np.logical_and(mask_lava_reached_repeated, mask_sampled_agent).reshape(size_batch, -1).any(-1)

    ratio_lava_reached = mask_samples_lava_reached.sum() / size_batch
    writer.add_scalar(f"{label}/ratio_lava_reached", ratio_lava_reached.item(), step_record)

    mask_goal_reached_repeated = np.repeat(mask_goal_reached[None, :], size_batch, 0)
    mask_samples_goal_reached = np.logical_and(mask_goal_reached_repeated, mask_sampled_agent).reshape(size_batch, -1).any(-1)

    # num_goals_per_sample = mask_sampled_goal.reshape(size_batch, -1).sum(-1)

    grid_x, grid_y = torch.meshgrid(
        torch.arange(obs_sampled_compact.shape[1], device=obs_sampled_compact.device),
        torch.arange(obs_sampled_compact.shape[2], device=obs_sampled_compact.device),
        indexing="ij",
    )
    grid_xy = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
    ijs = grid_xy.repeat(size_batch, 1, 1, 1)[slice_obs_sampled_compact == OBJECT_TO_IDX["agent"]]

    mask_obs_valid = np.zeros(size_batch, dtype=bool)

    env.load_layout_from_obs(obs_cond)
    for idx_sample in range(size_batch):
        ij = ijs[idx_sample]
        state = env.ijd2state(int(ij[0]), int(ij[1]))
        mask_obs_valid[idx_sample] = state in env.DP_info["states_reachable"]

    ratio_invalid_sample = 1 - mask_obs_valid.sum() / size_batch
    writer.add_scalar(f"{label}/ratio_invalid_sample", ratio_invalid_sample.item(), step_record)

    ratio_goal_reached = mask_samples_goal_reached.sum() / size_batch
    writer.add_scalar(f"{label}/ratio_goal_reached", ratio_goal_reached.item(), step_record)

    X = obs_sampled_compact.reshape(size_batch, -1)
    indices_unique = find_unique(X, dim=0)
    X_unique = X[indices_unique]

    writer.add_scalar(f"{label}/ratio_unique_samples", X_unique.shape[0] / size_batch, step_record)
    mask_unique = np.zeros(size_batch, dtype=bool)
    mask_unique[indices_unique] = True
    mask_valid_unique_sample = np.logical_and(mask_unique, mask_obs_valid)
    writer.add_scalar(f"{label}/ratio_valid_unique_samples", mask_valid_unique_sample.sum() / size_batch, step_record)

    obses_reconstructed = obs_sampled_compact.detach().to(torch.uint8).cpu().numpy()
    images = []
    for i in range(size_batch):
        image = env.render_obs(obses_reconstructed[i])
        if not mask_obs_valid[i]:
            image = 255 - image  # inverted color
        if mask_samples_lava_reached[i]:
            image = outline(image, color="red")
        elif mask_samples_goal_reached[i]:
            image = outline(image, color="green")
        images.append(image)
    writer.add_image(
        f"Vis_{label}/generated",
        gen_comparative_image(images, env.render_obs(obs_cond, tile_size=32)),
        step_record,
    )
    writer.flush()
    time_now = time.time()
    print(f"image batch generated at step {step_record} in {time_now - time_then:.2f}s")


def visualize_generation_minigrid(cvae, obs_cond, env, writer, step_record, suffix=""):
    layout_cond, _ = cvae.encoder(minigridobs2tensor(obs_cond))
    _, mask_agent_sampled = cvae.sample_from_uniform_prior(layout_cond)
    obs_sampled = cvae.decoder(layout_cond, mask_agent_sampled)
    debug_cvae_generation(obs_sampled, obs_cond, env, writer, step_record, label="Samples_All" + suffix)


def visualize_generation_minigrid2(cvae, obs_cond, env, writer, step_record, suffix=""):
    layout_cond, _ = cvae.layout_extractor(minigridobs2tensor(obs_cond))
    ret = cvae.sample_from_uniform_prior(minigridobs2tensor(obs_cond))
    if len(ret) == 2:
        _, mask_agent_sampled = ret
    else:
        mask_agent_sampled = ret
    obs_sampled = cvae.decoder(layout_cond, mask_agent_sampled)
    debug_cvae_generation(obs_sampled, obs_cond, env, writer, step_record, label="Samples_All" + suffix)


def cyclical_schedule(step, interval):
    assert step >= 0 and interval > 0
    step_local = step % interval
    half_interval = 0.5 * interval
    if step_local >= half_interval:
        return 1
    else:
        return step_local / half_interval


@torch.no_grad()
def find_unique(x, must_keep=[], dim=0):
    unique, inverse = torch.unique(x, sorted=False, return_inverse=True, dim=dim)
    X = inverse.reshape(-1, 1).detach().cpu().numpy()
    size_batch = X.shape[0]
    if len(must_keep):
        must_keep = np.array(must_keep)
        must_keep = np.arange(size_batch)[must_keep]  # NOTE(H): in case you are passing -1
        order = np.concatenate([must_keep, np.setdiff1d(np.arange(size_batch), must_keep)])
        X = X[order]
    else:
        order = np.arange(size_batch)
    size_must_keep = must_keep.size if len(must_keep) else 0
    mask_excluded = np.zeros(size_batch, dtype=bool)

    for idx_row in range(size_batch):
        if idx_row == size_batch - 1:
            continue
        if mask_excluded[idx_row]:
            continue
        else:
            X_curr = int(X[idx_row])
            mask_compare = np.logical_not(mask_excluded)
            mask_compare[0 : max(size_must_keep, idx_row + 1)] = False
            if not mask_compare.any():
                break
            mask_excluded[mask_compare] |= (X[mask_compare] == X_curr).all(-1)
    return np.sort(order[~mask_excluded]).tolist()


# @profile
@torch.no_grad()
def k_medoids(D, k, medoids_must_stay=[], max_iter=5):
    """Performs k-medoids clustering on the distance matrix D.

    Args:
        D (np.ndarray): Distance matrix of shape (n_samples, n_samples).
        k (int): Number of clusters.
        medoids_must_stay (list): Indices of points that must be chosen as medoids.
        max_iter (int): Maximum number of iterations.

    Returns:
        tuple: (M, C, cost) where M is a list of the indices of the final medoids and C is a list
            of the cluster assignments for each point.
    """
    n_samples = D.shape[0]
    device = D.device
    # Initialize medoids to be the partial medoids and random non-partial medoids
    initial_medoids = medoids_must_stay.copy()
    remaining_centers = list(set(range(n_samples)) - set(medoids_must_stay))
    initial_medoids = medoids_must_stay + np.random.choice(remaining_centers, size=k - len(initial_medoids), replace=False).tolist()
    remaining_centers_tensor = torch.tensor(remaining_centers, dtype=torch.long, device=device)
    M = torch.tensor(initial_medoids, dtype=torch.long, device=device)
    # Initialize cluster assignments and calculate the cost
    C = torch.argmin(D[:, M], dim=1)
    n_arange = torch.arange(n_samples, device=device)
    cost = torch.sum(D[n_arange, M[C]])
    new_medoids_template = torch.repeat_interleave(M[None, :], len(remaining_centers), dim=0)

    medoid_costs_template = torch.zeros(n_samples, device=device, dtype=D.dtype)
    medoid_costs_template[medoids_must_stay] = torch.inf

    for i in range(max_iter):  # Iterate over medoids
        improved = False
        for j in range(len(medoids_must_stay), k):  # If partial medoid, skip
            # Calculate cost of swapping medoid with each non-medoid point
            new_medoid_costs = medoid_costs_template.clone()
            new_medoids = new_medoids_template.clone()
            new_medoids[:, j] = remaining_centers_tensor
            D_new_medoids = D[:, new_medoids]
            argmins_D_new_medoids = torch.argmin(D_new_medoids, dim=-1)
            indices = new_medoids.gather(1, argmins_D_new_medoids.T)
            new_medoid_costs[remaining_centers_tensor] = torch.sum(D[n_arange, indices], dim=-1)
            # Select the new medoid that minimizes the cost
            new_medoid_idx = torch.argmin(new_medoid_costs)
            cost_low = new_medoid_costs[new_medoid_idx]
            new_M = M.clone()
            new_M[j] = new_medoid_idx
            # Update medoids and cost
            if cost_low < cost:
                # print(f"[iter {i:d}, cluster {j:d}]: cost dropped from {cost} to {cost_low}")
                M = new_M.clone()
                C = torch.argmin(D[:, M], dim=1)
                cost = cost_low
                improved = True
                new_medoids_template = torch.repeat_interleave(M[None, :], len(remaining_centers), dim=0)
        if not improved:
            break
    # print(f"{i + 1:d} ierations spent in total")
    return np.sort(M.cpu().numpy()).tolist(), C.cpu().numpy(), cost


def reachability_BFS(A, state_start, state_targ):  # NOTE(H): THIS ONLY WORKS FOR DETERMINISTIC CASE
    assert A.dtype == bool
    Q = queue.Queue()
    Q.put((state_start, 0))
    dist_reach = np.inf
    visited = np.zeros(A.shape[1], dtype=bool)
    while not Q.empty():
        state_curr, dist_curr = Q.get()
        if visited[state_curr]:
            continue
        else:
            visited[state_curr] = True
        if state_curr == state_targ:
            if np.isinf(dist_reach):
                dist_reach = 0
            dist_reach = dist_curr
            break
        states_reachable = np.where(A[state_curr])[0].tolist()
        for state_reachable in states_reachable:
            if state_reachable != state_curr:  # NOTE(H): no infinite loops
                Q.put((state_reachable, dist_curr + 1))
    return dist_reach
