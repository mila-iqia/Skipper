from typing import Dict, Callable, Optional, Iterable

import numpy as np

import scipy.spatial.distance as dist

from cpprb import ReplayBuffer, PrioritizedReplayBuffer, MPReplayBuffer, MPPrioritizedReplayBuffer


def unique_rows(X, threshold=0.1):
    size_batch = X.shape[0]
    dists = dist.cdist(X, X)
    excluded = []
    counter_unique = 0
    num_duplicates = (dists < threshold).sum(-1) - 1
    for idx_row in range(size_batch):
        if idx_row in excluded:
            continue
        counter_unique += 1
        if num_duplicates[idx_row] > 0:
            dists_ = dists[idx_row].squeeze()
            dists_[idx_row] = 10000
            excluded += np.where(dists_ < threshold)[0].tolist()
    mask_unique = np.ones(size_batch, dtype=bool)
    mask_unique[excluded] = False
    return X[mask_unique], excluded


class HindsightReplayBuffer:
    """
    Replay Buffer class for Hindsight Experience Replay (HER)

    Notes
    -----
    In Hindsight Experience Replay [1]_, failed transitions are considered
    as success transitions by re-labelling goal.

    References
    ----------
    .. [1] M. Andrychowicz et al, "Hindsight Experience Replay",
       Advances in Neural Information Processing Systems 30 (NIPS 2017),
       https://papers.nips.cc/paper/2017/hash/453fadbd8a1a3af50a9df4df899537b5-Abstract.html
       https://arxiv.org/abs/1707.01495
    """

    def __init__(
        self,
        size: int,
        env_dict: Dict,
        max_episode_len: int,
        reward_func: Callable,
        *,
        goal_func: Optional[Callable] = None,
        goal_shape: Optional[Iterable[int]] = None,
        state: str = "obs",
        action: str = "act",
        next_state: str = "next_obs",
        strategy: str = "future",
        additional_goals: int = 4,
        prioritized=True,
        num_goals_per_transition=1,
        unique_goals=False,
        new_logic=True,
        no_self_cycle=True,
        ctx=None,
        **kwargs,
    ):
        r"""
        Initialize ``HindsightReplayBuffer``

        Parameters
        ----------
        size : int
            Buffer Size
        env_dict : dict of dict
            Dictionary specifying environments. The keys of ``env_dict`` become
            environment names. The values of ``env_dict``, which are also ``dict``,
            defines ``"shape"`` (default ``1``) and ``"dtypes"`` (fallback to
            ``default_dtype``)
        max_episode_len : int
            Maximum episode length.
        reward_func : Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]
            Batch calculation of reward function:
            :math:`\mathcal{S}\times \mathcal{A}\times \mathcal{G} \to \mathcal{R}`.
        goal_func : Callable[[np.ndarray], np.ndarray], optional
            Batch extraction function for goal from state:
            :math:`\mathcal{S}\to\mathcal{G}`.
            If ``None`` (default), identity function is used (goal = state).
        goal_shape : Iterable[int], optional
            Shape of goal. If ``None`` (default), state shape is used.
        state : str, optional
            State name in ``env_dict``. The default is ``"obs"``.
        action : str, optional
            Action name in ``env_dict``. The default is ``"act"``.
        next_state : str, optional
            Next state name in ``env_dict``. The default is ``"next_obs"``.
        strategy : {"future", "episode", "random", "final"}, optional
            Goal sampling strategy.
            ``"future"`` selects one of the future states in the same episode.
            ``"episode"`` selects states in the same episode.
            ``"random"`` selects from the all states in replay buffer.
            ``"final"`` selects the final state in the episode.
            For ``"final"`` strategy, ``additional_goals`` is ignored.
            The default is ``"future"``.
        additional_goals : int, optional
            Number of additional goals. The default is ``4``.
        prioritized : bool, optional
            Whether use Prioritized Experience Replay. The default is ``True``.
        """
        self.max_episode_len = max_episode_len
        self.reward_func = reward_func
        self.goal_func = goal_func
        self.num_goals_per_transition = num_goals_per_transition
        self.unique_goals = unique_goals
        self.new_logic = new_logic
        self.no_self_cycle = no_self_cycle  # dont want target state to be the same state as what the current state is

        self.state = state
        self.action = action
        self.next_state = next_state

        self.strategy = strategy
        known_strategy = ["future", "episode", "random", "final"]
        if self.strategy not in known_strategy:
            raise ValueError(f"Unknown Strategy: {strategy}. " + f"Known Strategies: {known_strategy}")

        self.additional_goals = additional_goals
        if self.strategy == "final":
            self.additional_goals = 1

        self.prioritized = prioritized

        if goal_shape:
            goal_dict = {**env_dict[state], "shape": goal_shape}
            self.goal_shape = np.array(goal_shape, ndmin=1)
        else:
            goal_dict = env_dict[state]
            self.goal_shape = np.array(env_dict[state].get("shape", 1), ndmin=1)

        if self.reward_func is None:
            dict_init = {**env_dict, "goal": goal_dict}
        else:
            dict_init = {**env_dict, "rew_int": {}, "goal": goal_dict}
        for idx_num_goal in range(1, self.num_goals_per_transition):
            dict_init[f"goal{idx_num_goal}"] = goal_dict

        self.dict_rb_init = dict_init

        if ctx is not None:
            RB = ctx.PrioritizedReplayBuffer if self.prioritized else ctx.ReplayBuffer
        else:
            RB = PrioritizedReplayBuffer if self.prioritized else ReplayBuffer
        if self.prioritized and ctx is not None:
            self.rb = RB(size, self.dict_rb_init, check_for_update=False, **kwargs)
        else:
            self.rb = RB(size, self.dict_rb_init, **kwargs)
        if ctx is not None:
            self.episode_rb = ctx.ReplayBuffer(self.max_episode_len, env_dict)
        else:
            self.episode_rb = ReplayBuffer(self.max_episode_len, env_dict)

        self.rng = np.random.default_rng()

    def add(self, **kwargs):
        r"""Add transition(s) into replay buffer.

        Multple sets of transitions can be added simultaneously.

        Parameters
        ----------
        **kwargs : array like or float or int
            Transitions to be stored.
        """
        if self.episode_rb.get_stored_size() >= self.max_episode_len:
            raise ValueError("Exceed Max Episode Length")
        self.episode_rb.add(**kwargs)

    def sample(self, batch_size: int, **kwargs):
        r"""Sample the stored transitions randomly with specified size

        Parameters
        ----------
        batch_size : int
            sampled batch size

        Returns
        -------
        dict of ndarray
            Sampled batch transitions, which might contains
            the same transition multiple times.
        """
        return self.rb.sample(batch_size, **kwargs)

    def submit_to_rb(self, trajectory, transition, dict_goals):
        keys_dict_goal = dict_goals.keys()
        for idx_num_goal in range(1, self.num_goals_per_transition):
            key = f"goal{idx_num_goal}"
            if key not in keys_dict_goal:
                dict_goals[key] = dict_goals["goal"]

        if self.reward_func is None:
            self.rb.add(**transition, **dict_goals)
        else:
            rew_int = self.reward_func(transition[self.next_state], trajectory[self.action], dict_goals["goal"])
            self.rb.add(**transition, rew_int=rew_int, **dict_goals)

    def on_episode_end(self, goal=None):
        r"""
        Terminate the current episode and set hindsight goal

        Parameters
        ----------
        goal : array-like
            Original goal state of this episode.
        """
        episode_len = self.episode_rb.get_stored_size()
        if episode_len == 0:
            return None

        trajectory = self.episode_rb.get_all_transitions()
        possible_goals = trajectory[self.next_state]
        if self.unique_goals:
            possible_goals = unique_rows(possible_goals.reshape(possible_goals.shape[0], -1))[0].reshape(-1, *possible_goals.shape[1:])
        num_possible_goals = possible_goals.shape[0]
        if goal is not None:  # NOTE(H): if the goal is not specified
            assert self.num_goals_per_transition == 1
            add_shape = (trajectory[self.state].shape[0], *self.goal_shape)
            goal = np.broadcast_to(np.asarray(goal), add_shape)
            if self.reward_func is None:
                self.rb.add(**trajectory, goal=goal)
            else:
                rew_int = self.reward_func(possible_goals, trajectory[self.action], goal)

                self.rb.add(**trajectory, goal=goal, rew_int=rew_int)

        if self.strategy in ["episode", "future"]:
            idx = np.full(
                (
                    self.num_goals_per_transition * self.additional_goals,
                    num_possible_goals,
                ),
                -1,
                dtype=np.int64,
            )
            idx_processed_max = np.full((self.num_goals_per_transition, num_possible_goals), -1, dtype=np.int64)
            for i in range(num_possible_goals):
                low = 0 if self.strategy == "episode" else i  # i for future!
                # sort
                transition = {}
                for key in list(trajectory.keys()):
                    transition[key] = trajectory[key][i]
                if self.no_self_cycle:
                    coincidence = np.where((transition["obs"].reshape(-1) == possible_goals.reshape(num_possible_goals, -1)).all(-1))[0]
                    choices = np.setdiff1d(np.arange(low, num_possible_goals), coincidence)
                    if choices.shape[0] == 0:
                        continue  # NOTE: just ditch
                    idx[:, i] = np.sort(
                        self.rng.choice(
                            choices,
                            self.num_goals_per_transition * self.additional_goals,
                        )
                    )
                else:
                    idx[:, i] = np.sort(
                        self.rng.integers(
                            low=low,
                            high=num_possible_goals,
                            size=self.num_goals_per_transition * self.additional_goals,
                        )
                    )
                dict_goals = {}
                for j in range(self.num_goals_per_transition * self.additional_goals):
                    idx_targ = idx[j, i]
                    idx_num_goal = len(dict_goals)
                    if self.new_logic:
                        if idx_targ == -1 or idx_targ <= idx_processed_max[idx_num_goal, i]:
                            # NOTE: to recover the old logic of adding 4 goals no matter what, we just need to add stuffs here
                            if idx_num_goal and j == idx.shape[0] - 1:
                                self.submit_to_rb(trajectory, transition, dict_goals)
                        else:
                            idx_processed_max[idx_num_goal, i] = idx_targ
                            if self.goal_func is None:
                                goal = possible_goals[idx_targ]
                            else:
                                goal = self.goal_func(possible_goals[idx_targ])

                            if idx_num_goal == 0:
                                dict_goals["goal"] = goal
                            else:
                                dict_goals[f"goal{idx_num_goal}"] = goal
                            if idx_num_goal == self.num_goals_per_transition - 1:
                                self.submit_to_rb(trajectory, transition, dict_goals)
                                dict_goals = {}
                    else:
                        if self.goal_func is None:
                            goal = possible_goals[idx_targ]
                        else:
                            goal = self.goal_func(possible_goals[idx_targ])
                        if idx_num_goal == 0:
                            dict_goals["goal"] = goal
                        else:
                            dict_goals[f"goal{idx_num_goal}"] = goal
                        if idx_num_goal == self.num_goals_per_transition - 1:
                            self.submit_to_rb(trajectory, transition, dict_goals)
                            dict_goals = {}
        elif self.strategy == "final":
            assert self.num_goals_per_transition == 1
            if self.goal_func is None:
                goal = np.broadcast_to(possible_goals[-1], possible_goals.shape)
            else:
                goal = self.goal_func(np.broadcast_to(possible_goals[-1], possible_goals.shape))
            if self.reward_func is None:
                self.rb.add(**trajectory, goal=goal)
            else:
                rew_int = self.reward_func(possible_goals, trajectory[self.action], goal)
                self.rb.add(**trajectory, rew_int=rew_int, goal=goal)
        elif self.strategy == "random":  #
            # Note 1:
            #   We should not prioritize goal selection,
            #   so that we manually create indices.
            # Note 2:
            #   Since we cannot access internal data directly,
            #   we have to extract set of transitions.
            #   Although this has overhead, it is fine
            #   becaue "random" strategy is used only for
            #   strategy comparison.
            if self.num_goals_per_transition > 1:
                raise NotImplementedError("life is short and I am too busy")
            idx = self.rng.integers(
                low=0,
                high=self.rb.get_stored_size(),
                size=self.additional_goals * episode_len,
            )
            if self.goal_func is None:
                goal = self.rb._encode_sample(idx)[self.next_state]
            else:
                goal = self.goal_func(self.rb._encode_sample(idx)[self.next_state])
            goal = goal.reshape((self.additional_goals, episode_len, *(goal.shape[1:])))
            for g in goal:
                if self.reward_func is None:
                    self.rb.add(**trajectory, goal=g)
                else:
                    rew_int = self.reward_func(trajectory[self.next_state], trajectory[self.action], g)
                    self.rb.add(**trajectory, rew_int=rew_int, goal=g)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        self.episode_rb.clear()
        self.rb.on_episode_end()

    def clear(self):
        """
        Clear replay buffer
        """
        self.rb.clear()
        self.episode_rb.clear()

    def get_stored_size(self):
        """
        Get stored size

        Returns
        -------
        int
            stored size
        """
        return self.rb.get_stored_size()

    def get_buffer_size(self):
        """
        Get buffer size

        Returns
        -------
        int
            buffer size
        """
        return self.rb.get_buffer_size()

    def get_all_transitions(self, shuffle: bool = False):
        r"""
        Get all transitions stored in replay buffer.

        Parameters
        ----------
        shuffle : bool, optional
            When ``True``, transitions are shuffled. The default value is ``False``.

        Returns
        -------
        transitions : dict of numpy.ndarray
            All transitions stored in this replay buffer.
        """
        return self.rb.get_all_transitions(shuffle)

    def update_priorities(self, indexes, priorities):
        """
        Update priorities

        Parameters
        ----------
        indexes : array_like
            indexes to update priorities
        priorities : array_like
            priorities to update

        Raises
        ------
        TypeError: When ``indexes`` or ``priorities`` are ``None``
        ValueError: When this buffer is constructed with ``prioritized=False``
        """
        if not self.prioritized:
            raise ValueError("Buffer is constructed without PER")

        self.rb.update_priorities(indexes, priorities)

    def get_max_priority(self):
        """
        Get max priority

        Returns
        -------
        float
            Max priority of stored priorities

        Raises
        ------
        ValueError: When this buffer is constructed with ``prioritized=False``
        """
        if not self.prioritized:
            raise ValueError("Buffer is constructed without PER")

        return self.rb.get_max_priority()


class MPHindsightReplayBuffer(HindsightReplayBuffer):
    """
    Replay Buffer class for Hindsight Experience Replay (HER)

    Notes
    -----
    In Hindsight Experience Replay [1]_, failed transitions are considered
    as success transitions by re-labelling goal.

    References
    ----------
    .. [1] M. Andrychowicz et al, "Hindsight Experience Replay",
       Advances in Neural Information Processing Systems 30 (NIPS 2017),
       https://papers.nips.cc/paper/2017/hash/453fadbd8a1a3af50a9df4df899537b5-Abstract.html
       https://arxiv.org/abs/1707.01495
    """

    def __init__(
        self,
        size: int,
        env_dict: Dict,
        max_episode_len: int,
        reward_func: Callable,
        *,
        goal_func: Optional[Callable] = None,
        goal_shape: Optional[Iterable[int]] = None,
        state: str = "obs",
        action: str = "act",
        next_state: str = "next_obs",
        strategy: str = "future",
        additional_goals: int = 4,
        prioritized=True,
        num_goals_per_transition=1,
        ctx=None,
        **kwargs,
    ):
        r"""
        Initialize ``HindsightReplayBuffer``

        Parameters
        ----------
        size : int
            Buffer Size
        env_dict : dict of dict
            Dictionary specifying environments. The keys of ``env_dict`` become
            environment names. The values of ``env_dict``, which are also ``dict``,
            defines ``"shape"`` (default ``1``) and ``"dtypes"`` (fallback to
            ``default_dtype``)
        max_episode_len : int
            Maximum episode length.
        reward_func : Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]
            Batch calculation of reward function:
            :math:`\mathcal{S}\times \mathcal{A}\times \mathcal{G} \to \mathcal{R}`.
        goal_func : Callable[[np.ndarray], np.ndarray], optional
            Batch extraction function for goal from state:
            :math:`\mathcal{S}\to\mathcal{G}`.
            If ``None`` (default), identity function is used (goal = state).
        goal_shape : Iterable[int], optional
            Shape of goal. If ``None`` (default), state shape is used.
        state : str, optional
            State name in ``env_dict``. The default is ``"obs"``.
        action : str, optional
            Action name in ``env_dict``. The default is ``"act"``.
        next_state : str, optional
            Next state name in ``env_dict``. The default is ``"next_obs"``.
        strategy : {"future", "episode", "random", "final"}, optional
            Goal sampling strategy.
            ``"future"`` selects one of the future states in the same episode.
            ``"episode"`` selects states in the same episode.
            ``"random"`` selects from the all states in replay buffer.
            ``"final"`` selects the final state in the episode.
            For ``"final"`` strategy, ``additional_goals`` is ignored.
            The default is ``"future"``.
        additional_goals : int, optional
            Number of additional goals. The default is ``4``.
        prioritized : bool, optional
            Whether use Prioritized Experience Replay. The default is ``True``.
        """
        self.max_episode_len = max_episode_len
        self.reward_func = reward_func
        self.goal_func = goal_func
        self.num_goals_per_transition = num_goals_per_transition

        self.state = state
        self.action = action
        self.next_state = next_state

        self.strategy = strategy
        known_strategy = ["future", "episode", "random", "final"]
        if self.strategy not in known_strategy:
            raise ValueError(f"Unknown Strategy: {strategy}. " + f"Known Strategies: {known_strategy}")

        self.additional_goals = additional_goals
        if self.strategy == "final":
            self.additional_goals = 1

        self.prioritized = prioritized

        if goal_shape:
            goal_dict = {**env_dict[state], "shape": goal_shape}
            self.goal_shape = np.array(goal_shape, ndmin=1)
        else:
            goal_dict = env_dict[state]
            self.goal_shape = np.array(env_dict[state].get("shape", 1), ndmin=1)
        RB = MPPrioritizedReplayBuffer if self.prioritized else MPReplayBuffer
        if self.reward_func is None:
            dict_init = {**env_dict, "goal": goal_dict}
        else:
            dict_init = {**env_dict, "rew_int": {}, "goal": goal_dict}
        for idx_num_goal in range(1, self.num_goals_per_transition):
            dict_init[f"goal{idx_num_goal}"] = goal_dict

        self.dict_rb_init = dict_init
        self.rb = RB(size, self.dict_rb_init, ctx=ctx, **kwargs)

        self.episode_rb = MPReplayBuffer(self.max_episode_len, env_dict, ctx=ctx)

        self.rng = np.random.default_rng()
