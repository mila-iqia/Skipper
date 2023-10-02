"""
                        +x (dim 0)
                    0--------------→
                    |       3
                    |       ↑
                    |       |
    +y (dim 1)      |  2 ←--+--→ 0
                    |       |
                    |       ↓
                    ↓       1

width * height

"""

import numpy as np
import minigrid
from minigrid import *
from utils import dijkstra, floyd_warshall
import copy

from visual_utils import highlight_img2


class Grid(minigrid.Grid):
    """
    Represent a grid and operations on it
    """

    # Static cache of pre-renderer tiles
    tile_cache = {}

    def encode(self, vis_mask=None, ignore_color=False, ignore_dir=False):
        """
        Produce a compact numpy encoding of the grid
        """

        if vis_mask is None:
            vis_mask = np.ones((self.width, self.height), dtype=bool)

        array = np.zeros(
            (self.width, self.height, 3 - int(ignore_color) - int(ignore_dir)),
            dtype="uint8",
        )

        for i in range(self.width):
            for j in range(self.height):
                if vis_mask[i, j]:
                    v = self.get(i, j)

                    if v is None:
                        array[i, j, 0] = OBJECT_TO_IDX["empty"]
                        if not ignore_color:
                            array[i, j, 1] = 0
                        if not ignore_dir:
                            array[i, j, -1] = 0

                    else:
                        v_encoded = v.encode()
                        array[i, j, 0] = v_encoded[0]
                        if not ignore_color:
                            array[i, j, 1] = v_encoded[1]
                        if not ignore_dir:
                            array[i, j, -1] = v_encoded[-1]
        return array

    def render(self, tile_size, agent_pos, agent_dir=None, highlight_mask=None, obs=None):
        """
        Render this grid at a given scale
        :param r: target renderer object
        :param tile_size: tile size in pixels
        """

        if obs is None:
            width, height = self.width, self.height
        else:
            width, height = obs.shape[0], obs.shape[1]

        if highlight_mask is None:
            highlight_mask = np.zeros(shape=(width, height), dtype=bool)

        # Compute the total grid size
        width_px = width * tile_size
        height_px = height * tile_size

        img = np.zeros(shape=(height_px, width_px, 3), dtype=np.uint8)

        def obs2agentmap(obs, ignore_dir=False):
            slice = obs[:, :, 0]
            if ignore_dir:
                return slice == OBJECT_TO_IDX["agent"]
            else:
                return slice == OBJECT_TO_IDX["agent"], obs[:, :, -1]

        def obs2goalmap(obs):
            width, height, _ = obs.shape
            slice = obs[:, :, 0]
            return slice == OBJECT_TO_IDX["goal"]

        if obs is not None:
            lava_map = (obs[:, :, 0] == OBJECT_TO_IDX["lava"]).squeeze()
            agent_map, agent_dir_map = obs2agentmap(obs)  # NOTE(H): lots of agents potentially, lol
            goal_map = obs2goalmap(obs)

        # Render the grid
        for j in range(0, height):
            for i in range(0, width):
                if obs is None:
                    cell = self.get(i, j)
                else:
                    if lava_map[i, j]:
                        cell = Lava()
                    elif goal_map[i, j]:
                        cell = Goal()
                    else:
                        cell = None

                agent_here = agent_map[i, j]
                tile_img = self.render_tile(
                    cell,
                    agent_dir=agent_dir_map[i, j] if agent_here else None,
                    highlight=highlight_mask[i, j],
                    tile_size=tile_size,
                )

                ymin = j * tile_size
                ymax = (j + 1) * tile_size
                xmin = i * tile_size
                xmax = (i + 1) * tile_size
                img[ymin:ymax, xmin:xmax, :] = tile_img

        return img

    def render_tile(cls, obj, agent_dir=None, highlight=False, tile_size=TILE_PIXELS, subdivs=3):
        """
        Render a tile and cache the result
        """

        # Hash map lookup key for the cache
        key = (agent_dir, highlight, tile_size)
        key = obj.encode() + key if obj else key

        if key in cls.tile_cache:
            return cls.tile_cache[key]

        img = np.zeros(shape=(tile_size * subdivs, tile_size * subdivs, 3), dtype=np.uint8)

        # Draw the grid lines (top and left edges)
        fill_coords(img, point_in_rect(0, 0.031, 0, 1), (100, 100, 100))
        fill_coords(img, point_in_rect(0, 1, 0, 0.031), (100, 100, 100))

        if obj is not None:
            obj.render(img)

        # Overlay the agent on top
        if agent_dir is not None:
            tri_fn = point_in_triangle(
                (0.12, 0.19),
                (0.87, 0.50),
                (0.12, 0.81),
            )

            # Rotate the agent based on its direction
            tri_fn = rotate_fn(tri_fn, cx=0.5, cy=0.5, theta=0.5 * math.pi * agent_dir)
            fill_coords(img, tri_fn, (255, 0, 0))

        # Highlight the cell if needed
        if highlight:
            highlight_img2(img)

        # Downsample the image to perform supersampling/anti-aliasing
        img = downsample(img, subdivs)

        # Cache the rendered tile
        cls.tile_cache[key] = img

        return img


class RandDistShift(MiniGridEnv):
    def __init__(
        self,
        width=8,
        height=8,
        lava_density_range=[0.3, 0.4],
        min_num_route=1,
        gamma=0.99,
        ignore_color=False,
        uniform_init=False,
        stochasticity=0.0,
    ):
        lava_density = np.random.uniform(lava_density_range[0], lava_density_range[1])
        self.min_num_route = min_num_route
        self.transposed = False
        if self.transposed:
            self.total_possible_lava = width * height - 2 * height
        else:
            self.total_possible_lava = width * height - 2 * width
        self.max_lava_blocks = int(self.total_possible_lava * lava_density)
        self.agent_start_dir = np.random.randint(0, 4)
        if self.transposed:
            if np.random.rand() <= 0.5:
                self.agent_start_pos = (np.random.randint(1, width), 0)
                self.goal_pos = (np.random.randint(0, width - 1), height - 1)
            else:
                self.agent_start_pos = (np.random.randint(0, width), height - 1)
                self.goal_pos = (np.random.randint(0, width), 0)
        else:
            if np.random.rand() <= 0.5:
                self.agent_start_pos = (0, np.random.randint(0, height))
                self.goal_pos = (width - 1, np.random.randint(0, height))
            else:
                self.agent_start_pos = (width - 1, np.random.randint(0, height))
                self.goal_pos = (0, np.random.randint(0, height))

        self.rand_width = width
        self.rand_height = height
        self.ignore_color = bool(ignore_color)
        self.ignore_dir = False  # only v2 could change this for now
        self.generate_map()
        mission_space = MissionSpace(mission_func=lambda: "get to the green goal square")
        super().__init__(
            width=width,
            height=height,
            max_steps=64,
            see_through_walls=True,
            agent_view_size=int(2 * max(width, height) - 1),
            mission_space=mission_space,
        )
        self.gamma = gamma
        self.render_mode = "rgb_array"
        self.init_DP_info()
        self.uniform_init = uniform_init
        assert stochasticity >= 0.0 and stochasticity <= 1.0
        self.stochasticity = stochasticity

    def render_image(self, ijds):
        highlight_mask = np.zeros([self.width, self.height], dtype=bool)
        for idx_waypoint in range(ijds.shape[0]):
            ijd = ijds[idx_waypoint]
            highlight_mask[ijd[0], ijd[1]] = True
        rendered = np.flip(
            self.grid.render(32, self.agent_pos, self.agent_dir, highlight_mask=highlight_mask, obs=self.obs_curr),
            axis=0,
        )
        return rendered

    def load_layout_from_obs(self, obs):
        RandDistShift.check_obs_validity(obs)
        assert len(obs.shape) == 3
        width, height, _ = obs.shape
        assert self.width == width and self.height == height
        slice = obs[:, :, 0]
        self.lava_map = np.zeros_like(slice, dtype=bool)
        agent_pos, agent_dir = None, None
        self.goal_pos = None
        self.agent_start_dir = 0
        for i in range(width):
            for j in range(height):
                if slice[i, j] == OBJECT_TO_IDX["agent"]:
                    if self.ignore_dir:
                        agent_pos, agent_dir = (i, j), 0
                    else:
                        agent_pos, agent_dir = (i, j), int(obs[i, j, -1])
                    if obs[i, j, 1] == COLOR_TO_IDX["yellow"]:
                        self.lava_map[i, j] = True
                    elif obs[i, j, 1] == COLOR_TO_IDX["green"]:
                        self.goal_pos = (i, j)
                elif slice[i, j] == OBJECT_TO_IDX["goal"]:
                    self.goal_pos = (i, j)
                elif slice[i, j] == OBJECT_TO_IDX["lava"]:
                    self.lava_map[i, j] = True
        assert agent_pos is not None and self.goal_pos is not None
        if self.transposed:
            self.agent_start_pos = (np.random.randint(1, width), height - 1 - self.goal_pos[1])
        else:
            self.agent_start_pos = (width - 1 - self.goal_pos[0], np.random.randint(0, height))

        self._gen_grid(width, height)
        self.agent_pos, self.agent_dir = agent_pos, agent_dir
        self.init_DP_info()
        self.collect_states_reachable()
        self.obs_curr = self.gen_fullyobservable_obs()

    def init_DP_info(self):
        self.DP_info = {
            "goal_pos": self.goal_pos,
            "num_states": None,
            "lava_map": None,
            "Q_optimal": None,
            "r": None,
            "P": None,
            "A": None,
            "state_target_tuples": None,
            "obses_all": None,
            "obses_all_processed": None,
            "states_reachable": None,
        }

    def gen_fullyobservable_obs(self):
        return self.draw_obs_with_agent(self.agent_pos[0], self.agent_pos[1], self.agent_dir)

    def collect_states_reachable(self):
        if self.DP_info["lava_map"] is None:
            self.init_DP_assets()
        if self.DP_info["P"] is None:
            self.collect_transition_probs()
        if self.DP_info["A"] is None:
            self.collect_state_adjacency()
        i_agent, j_agent, d_agent = (
            self.agent_start_pos[0],
            self.agent_start_pos[1],
            self.agent_start_dir,
        )
        agent_state = self.ijd2state(i_agent, j_agent, d_agent)
        ret = dijkstra(self.DP_info["A"], agent_state)
        states_reachable = [agent_state]
        for target_state in range(len(ret)):
            distance = ret[target_state]
            if distance != np.inf and agent_state != target_state:
                states_reachable.append(target_state)
        self.DP_info["states_reachable"] = sorted(states_reachable)

    def generate_random_path(self, epsilon=0.35):
        goal = self.goal_pos
        current_state = np.array(self.agent_start_pos)
        duration = 0
        while True:
            if duration == 0:
                duration = np.random.randint(1, 4)
                difference_x, difference_y = (
                    goal[0] - current_state[0],
                    goal[1] - current_state[1],
                )
                x_rand, y_rand = False, False
                action_list, random_action_list = [], []
                if difference_x != 0:
                    direction_diff_x = int(np.sign(difference_x))
                    action_list.append([direction_diff_x, 0])
                    random_action_list.append([-direction_diff_x, 0])
                else:
                    random_action_list.append([np.random.randint(0, 1) * 2 - 1, 0])
                    x_rand = True

                if difference_y != 0:
                    direction_diff_y = int(np.sign(difference_y))
                    action_list.append([0, direction_diff_y])
                    random_action_list.append([0, -direction_diff_y])
                else:
                    random_action_list.append([0, np.random.randint(0, 1) * 2 - 1])
                    y_rand = True

            if np.random.uniform(0, 1) > epsilon:
                if len(action_list) == 0:
                    break
                else:
                    current_action = action_list[int(np.random.randint(0, len(action_list)))]
            else:
                if x_rand:
                    current_action = random_action_list[0]
                elif y_rand:
                    current_action = random_action_list[1]
                else:
                    current_action = random_action_list[int(np.random.randint(0, len(random_action_list)))]
            current_state[0] += current_action[0]
            current_state[1] += current_action[1]
            current_state[0] = np.clip(current_state[0], 0, self.rand_width - 1)
            current_state[1] = np.clip(current_state[1], 0, self.rand_height - 1)
            self.lava_map[current_state[0], current_state[1]] = False
            duration -= 1
            if current_state[0] == goal[0] and current_state[1] == goal[1]:
                break

    def reset_gen_map(self):
        self.lava_map = np.zeros((self.rand_width, self.rand_height), dtype=bool)
        if self.transposed:
            self.lava_map[0 : self.rand_width, 1 : self.rand_height - 1] = True
        else:
            self.lava_map[1 : self.rand_width - 1, 0 : self.rand_height] = True
        self.lava_map[self.agent_start_pos[0], self.agent_start_pos[1]] = False
        self.lava_map[self.goal_pos[0], self.goal_pos[1]] = False

    def generate_map(self):
        self.reset_gen_map()
        while True:
            for i in range(self.min_num_route):
                self.generate_random_path()
            remaining_lava_blocks = int(np.sum(self.lava_map))
            if remaining_lava_blocks > self.max_lava_blocks:
                break
            self.reset_gen_map()

        if remaining_lava_blocks > self.max_lava_blocks:
            lava_indices = np.nonzero(self.lava_map)
            lava_indices_x = lava_indices[0]
            lava_indices_y = lava_indices[1]
            perm = np.random.permutation(lava_indices_x.shape[0])
            lava_indices_x = lava_indices_x[perm]
            lava_indices_y = lava_indices_y[perm]
            for i in range(int(remaining_lava_blocks - self.max_lava_blocks)):
                self.lava_map[lava_indices_x[i], lava_indices_y[i]] = False

    def generate_obses_all(self):
        if self.DP_info["num_states"] is None:
            self.init_DP_assets()
        if self.DP_info["states_reachable"] is None:
            self.collect_states_reachable()
        self.DP_info["obses_all"] = self.state2obs(self.DP_info["states_reachable"])

    # @profile
    def generate_state_target_tuples(self, max_dist=16):
        if self.DP_info["lava_map"] is None:
            self.init_DP_assets()
        if self.DP_info["P"] is None:
            self.collect_transition_probs()
        if self.DP_info["A"] is None:
            self.collect_state_adjacency()
        if self.DP_info["states_reachable"] is None:
            self.collect_states_reachable()
        goal_i, goal_j = self.goal_pos

        tuples = []
        states_reachable = copy.copy(self.DP_info["states_reachable"])
        ijds_reachable = np.stack(self.state2ijd(states_reachable), 1)
        states_reachable_nonterminal = []
        mask_nonterminal_among_reachable = np.zeros(len(states_reachable), dtype=bool)
        for idx_state_reachable in range(len(states_reachable)):
            ijd = ijds_reachable[idx_state_reachable]
            i, j, d = ijd
            if self.DP_info["lava_map"][i, j] or i == goal_i and j == goal_j:
                continue  # dont bother if starting from lava or real goal
            else:
                states_reachable_nonterminal.append(states_reachable[idx_state_reachable])
                mask_nonterminal_among_reachable[idx_state_reachable] = True
        A_reduced = self.DP_info["A"][states_reachable_nonterminal, :][:, states_reachable_nonterminal]
        # start_ijds = ijds_reachable[mask_nonterminal_among_reachable]
        D = floyd_warshall(A_reduced)
        D[D > max_dist] = np.inf
        for ii in range(len(states_reachable_nonterminal)):
            for jj in range(len(states_reachable_nonterminal)):
                if ii == jj or D[ii, jj] >= max_dist:
                    continue
                tuples.append((states_reachable_nonterminal[ii], states_reachable_nonterminal[jj], int(D[ii, jj])))

        # A_reduced = self.DP_info["A"][states_reachable, :][:, states_reachable]
        # start_ijds = np.stack(self.state2ijd(states_reachable), 1)

        # tuples = []
        # for idx_start_state in range(len(states_reachable)):
        #     start_state = states_reachable[idx_start_state]
        #     start_ijd = start_ijds[idx_start_state]
        #     i, j, d = start_ijd
        #     if self.DP_info["lava_map"][i, j] or i == goal_i and j == goal_j:
        #         continue  # dont bother if starting from lava or real goal
        #     ret = dijkstra(A_reduced, idx_start_state, max_dist=max_dist)
        #     for idx_target_state in range(len(ret)):
        #         target_state = states_reachable[idx_target_state]
        #         distance = ret[idx_target_state]
        #         if distance < max_dist and start_state != target_state:
        #             tuples.append((start_state, target_state, distance))
        self.DP_info["state_target_tuples"] = tuples
        return tuples

    def gen_grid(self, width, height):
        self._gen_grid(width, height)

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        # self.grid.wall_rect(0, 0, width, height)

        # Place a goal square in the bottom-right corner
        self.put_obj(Goal(), self.goal_pos[0], self.goal_pos[1])

        for i in range(0, self.lava_map.shape[0]):
            for j in range(0, self.lava_map.shape[1]):
                if self.lava_map[i, j]:
                    self.grid.set(i, j, Lava())

        self.full_grid_base = self.grid.encode(ignore_color=self.ignore_color, ignore_dir=self.ignore_dir)

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = "get to the green goal square"

    def reset(self, same_init_pos=False):
        super().reset()
        if self.uniform_init and not same_init_pos:
            if self.DP_info["states_reachable"] is None:
                self.collect_states_reachable()
            while True:  # sample a random state in states_reachable and make sure it is not the goal state
                state_rand = int(np.random.choice(self.DP_info["states_reachable"]))
                i, j, d = self.state2ijd(state_rand)
                if not (i == self.goal_pos[0] and j == self.goal_pos[1]) and not self.DP_info["lava_map"][i, j]:
                    break
            self.agent_pos = (int(i), int(j))
            self.agent_dir = int(d)
        else:
            self.agent_pos = copy.copy(self.agent_start_pos)
            self.agent_dir = copy.copy(self.agent_start_dir)
        self.obs_curr = self.gen_fullyobservable_obs()
        self.obs_goal = self.draw_obs_with_agent(int(self.goal_pos[0]), int(self.goal_pos[1]), 0, lava_map=None)
        return self.obs_curr

    def check_inside(self, pos):
        flag_inside = True
        if pos[0] < 0 or pos[0] >= self.width:
            flag_inside = False
        if pos[1] < 0 or pos[1] >= self.height:
            flag_inside = False
        return flag_inside

    def move_forward(self):
        reward, done = 0.0, False
        fwd_pos = self.front_pos
        flag_inside = self.check_inside(fwd_pos)  # check if the tile in front is still inside the boundaries
        if flag_inside:
            fwd_cell = self.grid.get(*fwd_pos) if flag_inside else None
            if fwd_cell == None or fwd_cell.can_overlap():
                self.agent_pos = fwd_pos
            if fwd_cell != None:
                if fwd_cell.type == "goal":
                    done = True
                    reward = 1.0
                elif fwd_cell.type == "lava":
                    done = True
        return reward, done

    def obs2ijd(self, obs):
        if len(obs.shape) == 3:
            obs = obs[None, :, :, :]
        size_batch, width, height, _ = obs.shape
        slice_type = obs[:, :, :, 0]
        mask_agent = slice_type == OBJECT_TO_IDX["agent"]
        ret_i, ret_j, ret_d = [], [], []
        for idx_sample in range(size_batch):
            found = False
            for i in range(width):
                if found:
                    break
                for j in range(height):
                    if found:
                        break
                    elif mask_agent[idx_sample, i, j]:
                        found = True
                        ret_i.append(i)
                        ret_j.append(j)
                        if not self.ignore_dir:
                            ret_d.append(int(obs[idx_sample, i, j, -1]))
            if not found:
                raise RuntimeError("agent not found in given obs")
        assert len(ret_i) == len(ret_j)
        if not self.ignore_dir:
            assert len(ret_i) == len(ret_d)
        if len(ret_i) == 1:
            if self.ignore_dir:
                return ret_i[0], ret_j[0]
            else:
                return ret_i[0], ret_j[0], ret_d[0]
        else:
            agent_i = np.array(ret_i)
            agent_j = np.array(ret_j)
            if self.ignore_dir:
                return agent_i, agent_j
            else:
                agent_d = np.array(ret_d)
                return agent_i, agent_j, agent_d

    def obs2goalpos(self, obs=None):
        if obs is None:
            return self.goal_pos[0], self.goal_pos[1]
        else:
            width, height, _ = obs.shape
            slice = obs[:, :, 0]
            for i in range(width):
                for j in range(height):
                    if slice[i, j] == OBJECT_TO_IDX["goal"]:
                        return i, j
            raise RuntimeError("goal not found in given obs")

    def get_lava_map(self):
        maps = self.full_grid_base[:, :, 0] == OBJECT_TO_IDX["lava"]
        return maps.squeeze()

    @classmethod
    def check_obs_validity(cls, obs):
        if len(obs.shape) == 3:
            obs = obs[None, :, :, :]
        assert len(obs.shape) == 4
        slice_type = obs[:, :, :, 0]
        slice_color = obs[:, :, :, 1]
        mask_agents = slice_type == OBJECT_TO_IDX["agent"]
        num_agents = mask_agents.sum((-1, -2))
        mask_goals = slice_type == OBJECT_TO_IDX["goal"]
        num_goals = mask_goals.sum((-1, -2))
        assert (num_agents == 1).all()
        assert (num_goals <= 1).all() and (num_goals >= 0).all()
        mask_should_be_red_or_yellow = num_goals == 1
        mask_should_be_green = num_goals == 0
        colors_agent = slice_color[mask_agents]
        colors_agent_should_be_red_or_yellow = colors_agent[mask_should_be_red_or_yellow]
        colors_agent_should_be_green = colors_agent[mask_should_be_green]
        assert np.logical_or(
            colors_agent_should_be_red_or_yellow == COLOR_TO_IDX["red"],
            colors_agent_should_be_red_or_yellow == COLOR_TO_IDX["yellow"],
        ).all()
        assert (colors_agent_should_be_green == COLOR_TO_IDX["green"]).all()

    def obs2state(self, obs=None):
        if self.ignore_dir:
            agent_i, agent_j = self.obs2ijd(obs=obs)
            agent_d = np.zeros_like(agent_i)
        else:
            agent_i, agent_j, agent_d = self.obs2ijd(obs=obs)
        return self.ijd2state(agent_i, agent_j, agent_d)

    def obs2ijdstate(self, obs=None):
        if self.ignore_dir:
            agent_i, agent_j = self.obs2ijd(obs=obs)
            agent_d = np.zeros_like(agent_i)
        else:
            agent_i, agent_j, agent_d = self.obs2ijd(obs=obs)
        return self.ijd2state(agent_i, agent_j, agent_d), (agent_i, agent_j, agent_d)

    # @profile
    def generate_oracle(self, goal_pos=None):
        # Generate r
        self.init_DP_assets()
        r = self.collect_rewards(goal_pos=goal_pos)
        P = self.collect_transition_probs(goal_pos=goal_pos)

        VmulP = lambda v, P: np.matmul(P, v).transpose()
        v0 = np.zeros(self.num_states)
        Boper = lambda r, P, v: np.max(r + self.gamma * VmulP(v, P), axis=-1)
        v_old = v0
        while True:
            v_new = Boper(r, P, v_old)
            if np.sum(np.abs(v_new - v_old)) <= 1e-5:
                break
            v_old = v_new
        if goal_pos is not None:
            goal_i, goal_j = goal_pos
        goal_i_original, goal_j_original = self.goal_pos
        if goal_pos is None or goal_i == goal_i_original and goal_j == goal_j_original:  # original goal
            self.DP_info["goal_pos"] = self.goal_pos
            self.DP_info["Q_optimal"] = r + self.gamma * VmulP(v_new, P)
            self.DP_info["Q_optimal"].flags["WRITEABLE"] = False
            return self.DP_info
        else:
            DP_info = {
                "goal_pos": goal_pos,
                "num_states": self.DP_info["num_states"],
                "lava_map": self.DP_info["lava_map"],
                "Q_optimal": r + self.gamma * VmulP(v_new, P),
                "r": r,
                "P": P,
            }
            return DP_info

    # @profile
    def get_optimal_actions(self, state, DP_info=None, tol=1e-3):
        if DP_info is None:
            DP_info = self.DP_info
        q = DP_info["Q_optimal"][state, :].squeeze()
        q_max = np.max(q)
        return np.where(np.abs(q - q_max) < tol)[0].tolist()

    # @profile
    def evaluate_action(self, action, obs=None, goal_pos=None, DP_info=None):
        if obs is None:
            obs = self.obs_curr
        if DP_info is None:
            DP_info = self.DP_info
        if DP_info["Q_optimal"] is None:
            DP_info = self.generate_oracle(goal_pos=goal_pos)
        return float(action in self.get_optimal_actions(self.obs2state(obs), DP_info=DP_info))

    def draw_obs_with_agent(self, i, j, d, lava_map=None):
        if lava_map is None:
            if self.DP_info["lava_map"] is None:
                self.init_DP_assets()
            lava_map = self.DP_info["lava_map"]
        full_grid = np.copy(self.full_grid_base)
        full_grid[:, :, 1] = 0
        i, j, d = np.array(i).reshape(-1, 1), np.array(j).reshape(-1, 1), np.array(d).reshape(-1, 1)
        size_batch = i.size
        assert size_batch == j.size == d.size
        ijds = np.concatenate([i, j, d], 1)
        full_grid = np.repeat(full_grid[np.newaxis, :, :, :], size_batch, axis=0)
        for idx_sample in range(size_batch):
            _i, _j, _d = ijds[idx_sample].tolist()
            full_grid[idx_sample, _i, _j, 0] = OBJECT_TO_IDX["agent"]
            if not self.ignore_color:
                if lava_map[_i, _j]:
                    full_grid[idx_sample, _i, _j, 1] = COLOR_TO_IDX["yellow"]
                elif _i == self.goal_pos[0] and _j == self.goal_pos[1]:
                    full_grid[idx_sample, _i, _j, 1] = COLOR_TO_IDX["green"]
                else:
                    full_grid[idx_sample, _i, _j, 1] = COLOR_TO_IDX["red"]
            if not self.ignore_dir:
                full_grid[idx_sample, _i, _j, -1] = _d
        if full_grid.shape[0] == 1:
            full_grid = full_grid.squeeze(0)
        return full_grid

    def render_obs(self, obs, highlight=False, tile_size=32):
        return self.get_full_render(highlight, tile_size, obs=obs)

    def state2obs(self, state, return_ijd=False):
        i, j, d = self.state2ijd(state)
        obs = self.draw_obs_with_agent(i, j, d)
        RandDistShift.check_obs_validity(obs)
        if return_ijd:
            return obs, (i, j, d)
        else:
            return obs

    def ijd2obs(self, i, j, d=None):
        i, j = np.array(i), np.array(j)
        assert i.size == j.size
        if self.ignore_dir:
            d = np.zeros_like(i)
        else:
            assert d is not None and d.size == i.size
        obs = self.draw_obs_with_agent(i, j, d)
        RandDistShift.check_obs_validity(obs)
        return obs

    def init_DP_assets(self):
        raise NotImplementedError("implement in subclasses")

    def ijd2state(self, i, j, d):
        raise NotImplementedError("implement in subclasses")

    def collect_rewards(self):
        raise NotImplementedError("implement in subclasses")

    def collect_transition_probs(self):
        raise NotImplementedError("implement in subclasses")

    def collect_state_adjacency(self):
        if self.DP_info["A"] is None:
            if self.DP_info["P"] is None:
                self.collect_transition_probs()
            assert self.DP_info["P"] is not None
            self.DP_info["A"] = np.bitwise_or.reduce(self.DP_info["P"].astype(bool), axis=0)
        return self.DP_info["A"]

    def step(self, action):
        raise NotImplementedError("implement in subclasses")


# class RandDistShift1(RandDistShift):
#     """
#     W/ TURN-OR-FORWARD DYNAMICS
#     """

#     class Actions(IntEnum):
#         # Turn left, turn right, move forward
#         left = 0
#         right = 1
#         forward = 2

#     def __init__(
#         self,
#         width=8,
#         height=8,
#         lava_density_range=[0.3, 0.4],
#         min_num_route=1,
#         gamma=0.99,
#         ignore_color=False,
#         uniform_init=False,
#     ):
#         super().__init__(
#             width=width,
#             height=height,
#             lava_density_range=lava_density_range,
#             min_num_route=min_num_route,
#             gamma=gamma,
#             ignore_color=ignore_color,
#             uniform_init=uniform_init,
#         )
#         self.actions = RandDistShift1.Actions
#         self.num_actions = len(self.actions)
#         self.action_space = spaces.Discrete(self.num_actions)
#         self.gamma = gamma

#         self.observation_space = spaces.Box(
#             low=0,
#             high=255,
#             shape=(self.width, self.height, 3 - int(ignore_color)),  # number of cells
#             dtype="uint8",
#         )
#         self.obs_curr = self.reset()

#     def init_DP_assets(self):
#         self.num_states = self.width * self.height * 4
#         self.DP_info["num_states"] = self.num_states
#         self.DP_info["lava_map"] = self.get_lava_map()

#     def ijd2state(self, i, j, d):
#         assert i.size == j.size == d.size
#         assert i >= 0 and i < self.width
#         assert j >= 0 and j < self.height
#         assert d >= 0 and d < 4
#         return i * 4 * self.width + j * 4 + d

#     def state2ijd(self, state):
#         state = np.array(state)
#         i = state // (4 * self.width)
#         j = (state - i * 4 * self.width) // 4
#         d = state - i * 4 * self.width - j * 4
#         assert (i >= 0).all() and (i < self.width).all()
#         assert (j >= 0).all() and (j < self.height).all()
#         assert (d >= 0).all() and (d < 4).all()
#         return i, j, d

#     def collect_rewards(self, goal_pos=None):
#         if goal_pos is None:
#             original_goal = True
#             goal_pos = self.goal_pos
#         else:
#             original_goal = False
#         goal_i, goal_j = goal_pos
#         goal_i_original, goal_j_original = self.goal_pos
#         r = np.zeros([self.num_states, self.num_actions])
#         if goal_j < self.height - 1 and not self.DP_info["lava_map"][goal_i, goal_j + 1] and not (goal_i == goal_i_original and goal_j + 1 == goal_j_original):
#             r[self.ijd2state(goal_i, goal_j + 1, 3), self.actions.forward] = 1
#         if goal_i < self.width - 1 and not self.DP_info["lava_map"][goal_i + 1, goal_j] and not (goal_i + 1 == goal_i_original and goal_j == goal_j_original):
#             r[self.ijd2state(goal_i + 1, goal_j, 2), self.actions.forward] = 1
#         if goal_j > 0 and not self.DP_info["lava_map"][goal_i, goal_j - 1] and not (goal_i == goal_i_original and goal_j - 1 == goal_j_original):
#             r[self.ijd2state(goal_i, goal_j - 1, 1), self.actions.forward] = 1
#         if goal_i > 0 and not self.DP_info["lava_map"][goal_i - 1, goal_j] and not (goal_i - 1 == goal_i_original and goal_j == goal_j_original):
#             r[self.ijd2state(goal_i - 1, goal_j, 0), self.actions.forward] = 1
#         if original_goal:
#             self.DP_info["r"] = r
#             self.DP_info["r"].flags["WRITEABLE"] = False
#         return r

#     def collect_transition_probs(self, goal_pos=None):
#         if goal_pos is None:
#             original_goal = True
#             goal_pos = self.goal_pos
#         else:
#             original_goal = False
#         if self.DP_info["P"] is None:
#             goal_i_original, goal_j_original = self.goal_pos
#             P = np.zeros([self.num_actions, self.num_states, self.num_states], dtype=np.float32)
#             for i in range(self.width):
#                 for j in range(self.height):
#                     for d in range(4):
#                         idx_state = self.ijd2state(i, j, d)
#                         if goal_i_original == i and goal_j_original == j or self.DP_info["lava_map"][i, j]:
#                             P[:, idx_state, idx_state] = 1.0
#                             continue

#                         P[0, idx_state, self.ijd2state(i, j, ((d - 1) % 4))] = 1  # turn left
#                         P[1, idx_state, self.ijd2state(i, j, ((d + 1) % 4))] = 1  # turn right

#                         if d == 1:
#                             P[2, idx_state, self.ijd2state(i, min(j + 1, self.height - 1), d)] = 1
#                         elif d == 0:
#                             P[2, idx_state, self.ijd2state(min(i + 1, self.width - 1), j, d)] = 1
#                         elif d == 3:
#                             P[2, idx_state, self.ijd2state(i, max(j - 1, 0), d)] = 1
#                         elif d == 2:
#                             P[2, idx_state, self.ijd2state(max(i - 1, 0), j, d)] = 1
#             self.DP_info["P"] = P
#             self.DP_info["P"].flags["WRITEABLE"] = False
#         if original_goal:
#             return self.DP_info["P"]
#         else:
#             goal_i, goal_j = goal_pos
#             P = np.copy(self.DP_info["P"])
#             P.flags["WRITEABLE"] = True
#             for d in range(4):
#                 idx_state = self.ijd2state(goal_i, goal_j, d)
#                 P[:, idx_state, :] = 0.0
#                 P[:, idx_state, idx_state] = 1.0
#             return P

#     # def evaluate_action_extra(self, obs=None):
#     #     if obs is None: obs = self.obs_curr
#     #     if not len(self.DP_info): self.generate_oracle()
#     #     list_quality_actions = []
#     #     q = self.DP_info["Q_optimal"][self.obs2state(obs), :].squeeze()
#     #     q_max = np.max(q)
#     #     list_optimal_actions = np.where(q == q_max)[0].tolist()
#     #     for action in range(self.action_space.n):
#     #         list_quality_actions.append(float(action in list_optimal_actions))
#     #     return list_quality_actions, q

#     def step(self, action):
#         self.step_count += 1
#         reward = 0.0
#         done, overtime = False, False
#         if action == self.actions.left:
#             self.agent_dir = (self.agent_dir - 1) % 4
#         elif action == self.actions.right:
#             self.agent_dir = (self.agent_dir + 1) % 4
#         elif action == self.actions.forward:
#             reward, done = self.move_forward()
#         else:
#             raise RuntimeError("unknown action")
#         if self.step_count >= self.max_steps:
#             done, overtime = True, True
#         aux = {"overtime": overtime}
#         if done:
#             aux["agent_pos"] = [self.agent_pos[0], self.agent_pos[1]]
#             aux["agent_dir"] = int(self.agent_dir)
#             aux["agent_pos_init"] = self.agent_start_pos
#             aux["dist2init"] = int(np.abs(self.agent_pos[0] - self.agent_start_pos[0]) + np.abs(self.agent_pos[1] - self.agent_start_pos[1]))
#             aux["dist2goal"] = int(np.abs(self.agent_pos[0] - self.goal_pos[0]) + np.abs(self.agent_pos[1] - self.goal_pos[1]))
#         self.obs_curr = self.gen_fullyobservable_obs()
#         return self.obs_curr, reward, done, aux


class RandDistShift2(RandDistShift):
    """
    W/ DIRECTIONAL-FORWARD DYNAMICS
    """

    class Actions(IntEnum):
        east = 0  # x+
        south = 1  # y+
        west = 2  # x-
        north = 3  # y-

    def __init__(
        self,
        width=8,
        height=8,
        lava_density_range=[0.3, 0.4],
        min_num_route=1,
        gamma=0.99,
        ignore_color=False,
        ignore_dir=True,
        uniform_init=False,
        stochasticity=0.0,
    ):
        super().__init__(
            width=width,
            height=height,
            lava_density_range=lava_density_range,
            min_num_route=min_num_route,
            gamma=gamma,
            ignore_color=ignore_color,
            uniform_init=uniform_init,
            stochasticity=stochasticity,
        )
        self.actions = RandDistShift2.Actions
        self.num_actions = len(self.actions)
        self.action_space = spaces.Discrete(self.num_actions)
        self.gamma = gamma
        self.ignore_dir = bool(ignore_dir)

        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(
                self.width,
                self.height,
                3 - int(self.ignore_dir) - int(ignore_color),
            ),  # number of cells
            dtype="uint8",
        )
        self.obs_curr = self.reset()

    def init_DP_assets(self):
        self.num_states = self.width * self.height
        self.DP_info["num_states"] = self.num_states
        self.DP_info["lava_map"] = self.get_lava_map()

    def collect_rewards(self, goal_pos=None):
        if goal_pos is None:
            original_goal = True
            goal_pos = self.goal_pos
        else:
            original_goal = False
        goal_i, goal_j = goal_pos
        goal_i_original, goal_j_original = self.goal_pos
        r = np.zeros([self.num_states, self.num_actions])
        if goal_j != self.height - 1 and not self.DP_info["lava_map"][goal_i, goal_j + 1] and not (goal_i == goal_i_original and goal_j + 1 == goal_j_original):
            r[self.ijd2state(goal_i, goal_j + 1), self.actions.north] = 1
        if goal_i != self.width - 1 and not self.DP_info["lava_map"][goal_i + 1, goal_j] and not (goal_i + 1 == goal_i_original and goal_j == goal_j_original):
            r[self.ijd2state(goal_i + 1, goal_j), self.actions.west] = 1
        if goal_j != 0 and not self.DP_info["lava_map"][goal_i, goal_j - 1] and not (goal_i == goal_i_original and goal_j - 1 == goal_j_original):
            r[self.ijd2state(goal_i, goal_j - 1), self.actions.south] = 1
        if goal_i != 0 and not self.DP_info["lava_map"][goal_i - 1, goal_j] and not (goal_i - 1 == goal_i_original and goal_j == goal_j_original):
            r[self.ijd2state(goal_i - 1, goal_j), self.actions.east] = 1
        if original_goal:
            self.DP_info["r"] = r
            self.DP_info["r"].flags["WRITEABLE"] = False
        return r

    def ijd2state(self, i, j, d=None):
        i, j = np.array(i), np.array(j)
        if d is not None:
            d = np.array(d)
        assert i.size == j.size
        if d is not None:
            assert i.size == d.size
        assert (i >= 0).all() and (i < self.width).all()
        assert (j >= 0).all() and (j < self.height).all()
        return i * self.width + j

    def state2ijd(self, state):
        state = np.array(state)
        i = state // self.width
        j = state - i * self.width
        assert i.size == j.size == state.size
        assert (i >= 0).all() and (i < self.width).all()
        assert (j >= 0).all() and (j < self.height).all()
        d = np.zeros_like(state)
        return i, j, d

    def collect_transition_probs(self, goal_pos=None):
        if goal_pos is None:
            original_goal = True
            goal_pos = self.goal_pos
        else:
            original_goal = False
        if self.DP_info["P"] is None:
            goal_i_original, goal_j_original = self.goal_pos
            P = np.zeros([self.num_actions, self.num_states, self.num_states], dtype=np.float32)
            for i in range(self.width):
                for j in range(self.height):
                    idx_state = self.ijd2state(i, j)
                    if goal_i_original == i and goal_j_original == j or self.DP_info["lava_map"][i, j]:
                        P[:, idx_state, idx_state] = 1.0
                        continue
                    for a in self.actions:
                        dx, dy = DIR_TO_VEC[a]
                        dx, dy = int(dx), int(dy)
                        i_next, j_next = max(0, min(self.width - 1, dx + i)), max(0, min(self.height - 1, dy + j))
                        idx_state_next = self.ijd2state(i_next, j_next)
                        P[a, idx_state, idx_state_next] = 1.0
            self.DP_info["P"] = P
            self.DP_info["P"].flags["WRITEABLE"] = False
        if original_goal:
            return self.DP_info["P"]
        else:
            goal_i, goal_j = goal_pos
            P = np.copy(self.DP_info["P"])
            P.flags["WRITEABLE"] = True
            idx_state = self.ijd2state(goal_i, goal_j)
            P[:, idx_state, :] = 0.0
            P[:, idx_state, idx_state] = 1.0
            return P

    def step(self, action):
        self.step_count += 1
        reward = 0.0
        done, overtime = False, False
        if self.stochasticity > 0:
            if np.random.rand() < self.stochasticity:
                action = np.random.randint(self.num_actions)
        self.agent_dir = action  # NOTE(H): this assumes the alignment of directions and actions
        reward, done = self.move_forward()
        if self.step_count >= self.max_steps:
            done, overtime = True, True
        self.agent_dir = 0
        aux = {"overtime": overtime}
        if done:
            aux["agent_pos"] = [self.agent_pos[0], self.agent_pos[1]]
            aux["agent_dir"] = int(self.agent_dir)
            aux["agent_pos_init"] = self.agent_start_pos
            aux["dist2init"] = int(np.abs(self.agent_pos[0] - self.agent_start_pos[0]) + np.abs(self.agent_pos[1] - self.agent_start_pos[1]))
            aux["dist2goal"] = int(np.abs(self.agent_pos[0] - self.goal_pos[0]) + np.abs(self.agent_pos[1] - self.goal_pos[1]))
        self.obs_curr = self.gen_fullyobservable_obs()
        return self.obs_curr, reward, done, aux


# class RandDistShift3(RandDistShift):
#     """
#     W/ TURN-AND-FORWARD DYNAMICS
#     """

#     class Actions(IntEnum):
#         left_forward = 0
#         forward = 1
#         right_forward = 2
#         back_forward = 3

#     def __init__(
#         self,
#         width=8,
#         height=8,
#         lava_density_range=[0.3, 0.4],
#         min_num_route=1,
#         gamma=0.99,
#         ignore_color=False,
#         uniform_init=False,
#     ):
#         super().__init__(
#             width=width,
#             height=height,
#             lava_density_range=lava_density_range,
#             min_num_route=min_num_route,
#             gamma=gamma,
#             ignore_color=ignore_color,
#             uniform_init=uniform_init,
#         )
#         self.actions = RandDistShift3.Actions
#         self.num_actions = len(self.actions)
#         self.action_space = spaces.Discrete(self.num_actions)
#         self.gamma = gamma

#         self.observation_space = spaces.Box(
#             low=0,
#             high=255,
#             shape=(self.width, self.height, 3 - int(ignore_color)),  # number of cells
#             dtype="uint8",
#         )
#         self.obs_curr = self.reset()

#     def init_DP_assets(self):
#         self.num_states = self.width * self.height * 4
#         self.DP_info["num_states"] = self.num_states
#         self.DP_info["lava_map"] = self.get_lava_map()

#     def ijd2state(self, i, j, d):
#         i, j, d = np.array(i), np.array(j), np.array(d)
#         assert i.size == j.size == d.size
#         assert (i >= 0).all() and (i < self.width).all()
#         assert (j >= 0).all() and (j < self.height).all()
#         assert (d >= 0).all() and (d < 4).all()
#         return i * 4 * self.width + j * 4 + d

#     def state2ijd(self, state):
#         state = np.array(state)
#         i = state // (4 * self.width)
#         j = (state - i * 4 * self.width) // 4
#         d = state - i * 4 * self.width - j * 4
#         assert (i >= 0).all() and (i < self.width).all()
#         assert (j >= 0).all() and (j < self.height).all()
#         assert (d >= 0).all() and (d < 4).all()
#         return i, j, d

#     def collect_rewards(self, goal_pos=None):
#         if goal_pos is None:
#             original_goal = True
#             goal_pos = self.goal_pos
#         else:
#             original_goal = False
#         goal_i, goal_j = goal_pos
#         goal_i_original, goal_j_original = self.goal_pos
#         r = np.zeros([self.num_states, self.num_actions])
#         if goal_j != self.height - 1 and not self.DP_info["lava_map"][goal_i, goal_j + 1] and not (goal_i == goal_i_original and goal_j + 1 == goal_j_original):
#             r[self.ijd2state(goal_i, goal_j + 1, 3), self.actions.forward] = 1
#             r[self.ijd2state(goal_i, goal_j + 1, 2), self.actions.right_forward] = 1
#             r[self.ijd2state(goal_i, goal_j + 1, 1), self.actions.back_forward] = 1
#             r[self.ijd2state(goal_i, goal_j + 1, 0), self.actions.left_forward] = 1
#         if goal_i != self.width - 1 and not self.DP_info["lava_map"][goal_i + 1, goal_j] and not (goal_i + 1 == goal_i_original and goal_j == goal_j_original):
#             r[self.ijd2state(goal_i + 1, goal_j, 3), self.actions.left_forward] = 1
#             r[self.ijd2state(goal_i + 1, goal_j, 2), self.actions.forward] = 1
#             r[self.ijd2state(goal_i + 1, goal_j, 1), self.actions.right_forward] = 1
#             r[self.ijd2state(goal_i + 1, goal_j, 0), self.actions.back_forward] = 1
#         if goal_j != 0 and not self.DP_info["lava_map"][goal_i, goal_j - 1] and not (goal_i == goal_i_original and goal_j - 1 == goal_j_original):
#             r[self.ijd2state(goal_i, goal_j - 1, 3), self.actions.back_forward] = 1
#             r[self.ijd2state(goal_i, goal_j - 1, 2), self.actions.left_forward] = 1
#             r[self.ijd2state(goal_i, goal_j - 1, 1), self.actions.forward] = 1
#             r[self.ijd2state(goal_i, goal_j - 1, 0), self.actions.right_forward] = 1
#         if goal_i != 0 and not self.DP_info["lava_map"][goal_i - 1, goal_j] and not (goal_i - 1 == goal_i_original and goal_j == goal_j_original):
#             r[self.ijd2state(goal_i - 1, goal_j, 3), self.actions.right_forward] = 1
#             r[self.ijd2state(goal_i - 1, goal_j, 2), self.actions.back_forward] = 1
#             r[self.ijd2state(goal_i - 1, goal_j, 1), self.actions.left_forward] = 1
#             r[self.ijd2state(goal_i - 1, goal_j, 0), self.actions.forward] = 1
#         if original_goal:
#             self.DP_info["r"] = r
#             self.DP_info["r"].flags["WRITEABLE"] = False
#         return r

#     def collect_transition_probs(self, goal_pos=None):
#         if goal_pos is None:
#             original_goal = True
#             goal_pos = self.goal_pos
#         else:
#             original_goal = False
#         if self.DP_info["P"] is None:
#             goal_i_original, goal_j_original = self.goal_pos
#             P = np.zeros([self.num_actions, self.num_states, self.num_states], dtype=np.float32)
#             for i in range(self.width):
#                 for j in range(self.height):
#                     for d in range(4):
#                         idx_state = self.ijd2state(i, j, d)
#                         if i == goal_i_original and j == goal_j_original or self.DP_info["lava_map"][i, j]:
#                             P[:, idx_state, idx_state] = 1.0
#                             continue

#                         for a in self.actions:
#                             d_next = self.new_dir(d, a)
#                             dx, dy = DIR_TO_VEC[d_next]
#                             dx, dy = int(dx), int(dy)
#                             i_next, j_next = max(0, min(self.width - 1, dx + i)), max(0, min(self.height - 1, dy + j))
#                             idx_state_next = self.ijd2state(i_next, j_next, d_next)
#                             P[a, idx_state, idx_state_next] = 1.0
#             self.DP_info["P"] = P
#             self.DP_info["P"].flags["WRITEABLE"] = False
#         if original_goal:
#             return self.DP_info["P"]
#         else:
#             goal_i, goal_j = goal_pos
#             P = np.copy(self.DP_info["P"])
#             P.flags["WRITEABLE"] = True
#             for d in range(4):
#                 idx_state = self.ijd2state(goal_i, goal_j, d)
#                 P[:, idx_state, :] = 0.0
#                 P[:, idx_state, idx_state] = 1.0
#             return P

#     def new_dir(self, dir_curr, action):
#         if action == self.actions.left_forward:
#             dir_next = (dir_curr - 1) % 4
#         elif action == self.actions.right_forward:
#             dir_next = (dir_curr + 1) % 4
#         elif action == self.actions.back_forward:
#             dir_next = (dir_curr + 2) % 4
#         elif action == self.actions.forward:
#             dir_next = dir_curr
#         else:
#             raise RuntimeError("unknown action")
#         return dir_next

#     def step(self, action):
#         self.step_count += 1
#         reward = 0
#         done, overtime = False, False
#         self.agent_dir = self.new_dir(self.agent_dir, action)
#         reward, done = self.move_forward()
#         if self.step_count >= self.max_steps:
#             done, overtime = True, True
#         aux = {"overtime": overtime}
#         if done:
#             aux["agent_pos"] = [self.agent_pos[0], self.agent_pos[1]]
#             aux["agent_dir"] = int(self.agent_dir)
#             aux["agent_pos_init"] = self.agent_start_pos
#             aux["dist2init"] = int(np.abs(self.agent_pos[0] - self.agent_start_pos[0]) + np.abs(self.agent_pos[1] - self.agent_start_pos[1]))
#             aux["dist2goal"] = int(np.abs(self.agent_pos[0] - self.goal_pos[0]) + np.abs(self.agent_pos[1] - self.goal_pos[1]))
#         self.obs_curr = self.gen_fullyobservable_obs()
#         return self.obs_curr, reward, done, aux
