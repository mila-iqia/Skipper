import time, datetime, numpy as np, os, pickle, random
from gym.envs.registration import register as gym_register

gym_register(id="RandDistShift-v1", entry_point="RandDistShift:RandDistShift1", reward_threshold=0.95)
gym_register(id="RandDistShift-v2", entry_point="RandDistShift:RandDistShift2", reward_threshold=0.95)
gym_register(id="RandDistShift-v3", entry_point="RandDistShift:RandDistShift3", reward_threshold=0.95)

from leap_utils import create_leap_agent

from tensorboardX import SummaryWriter
from runtime import generate_exptag, get_set_seed, get_new_env, config_parser, save_code_snapshot, evaluate_agent

# import line_profiler
# profile = line_profiler.LineProfiler()

parser = config_parser(mp=False)
args = parser.parse_args()

config_train = {
    "size": args.size_world,
    "gamma": args.gamma,
    "lava_density_range": [0.4, 0.4],
    "uniform_init": bool(args.uniform_init),
    "stochasticity": args.stochasticity,
}

configs_eval = [
    {
        "size": args.size_world,
        "gamma": args.gamma,
        "lava_density_range": [0.2, 0.3],
        "uniform_init": False,
        "stochasticity": args.stochasticity,
    },
    {
        "size": args.size_world,
        "gamma": args.gamma,
        "lava_density_range": [0.3, 0.4],
        "uniform_init": False,
        "stochasticity": args.stochasticity,
    },
    {
        "size": args.size_world,
        "gamma": args.gamma,
        "lava_density_range": [0.4, 0.5],
        "uniform_init": False,
        "stochasticity": args.stochasticity,
    },
    {
        "size": args.size_world,
        "gamma": args.gamma,
        "lava_density_range": [0.5, 0.6],
        "uniform_init": False,
        "stochasticity": args.stochasticity,
    },
]

envs_train = []

env = get_new_env(args, **config_train)

args.seed_rl_run = random.randint(0, 1000000)
args.leap_vae_discrete = True
assert len(args.seed), "must load vae checkpoint"
args.seed = get_set_seed(args.seed, env)
args.method = "leap"
args.num_waypoints = 5
args.suppress_delusion = True
args = generate_exptag(args, additional="")

if args.random_walk_leap:
    path_tf_events = f"tb_records/{env.spec.id}/{args.size_world}x{args.size_world}/leap/rl_pretrain/{args.comments}/from{args.seed}_RW/{args.seed_rl_run}"
else:
    path_tf_events = f"tb_records/{env.spec.id}/{args.size_world}x{args.size_world}/leap/rl_pretrain/{args.comments}/from{args.seed}/{args.seed_rl_run}"

if args.uniform_init:
    folder_checkpoints = f"tb_records/{env.spec.id}/{args.size_world}x{args.size_world}/leap/vae_discrete_pretrain/{args.comments}/{args.seed}"
else:
    folder_checkpoints = f"tb_records/{env.spec.id}/{args.size_world}x{args.size_world}/leap/vae_discrete_pretrain_non_uniform/{args.comments}/{args.seed}"

writer = SummaryWriter(path_tf_events)
args.path_pretrained_vae = os.path.join(folder_checkpoints, "cvae_leap.pt")
args.path_pretrain_envs = os.path.join(folder_checkpoints, "envs.pkl")

if args.num_envs_train:
    with open(args.path_pretrain_envs, "rb") as file:
        envs_train = pickle.load(file)

if args.num_envs_train > 0:
    assert len(envs_train) == args.num_envs_train

    def generator_env_train():
        idx_env = np.random.randint(args.num_envs_train)
        return envs_train[idx_env]

else:

    def generator_env_train():
        env_train = get_new_env(args, **config_train)
        return env_train


save_code_snapshot(path_tf_events)

print(args)
agent = create_leap_agent(args, env=env, dim_embed=args.dim_embed, num_actions=env.action_space.n)
# TODO(H): load the envs too
################################################################

milestones_evaluation = []
step_milestone, pointer_milestone = 0, 0
while step_milestone <= args.steps_stop:
    milestones_evaluation.append(step_milestone)
    step_milestone += args.freq_eval

episode_elapsed = 0
time_start = time.time()
return_cum, return_cum_discount, steps_episode, time_episode_start, str_info = 0.0, 0.0, 0, time.time(), ""

# TODO(H): what if the state_rep collapses? is the only way to do this to be inside an agent?
while True:
    if args.randomized:
        env = generator_env_train()
    obs_curr, done = env.reset(same_init_pos=False), False
    if not args.disable_eval and pointer_milestone < len(milestones_evaluation) and agent.steps_interact >= milestones_evaluation[pointer_milestone]:
        env_generator = lambda: generator_env_train()
        returns_mean, returns_std, returns_discounted_mean, returns_discounted_std = evaluate_agent(env_generator, agent, num_episodes=20, type_env="minigrid")
        print(
            f"Eval/train x{20} @ step {agent.steps_interact:d} - returns_mean: {returns_mean:.2f}, returns_std: {returns_std:.2f}, returns_discounted_mean: {returns_discounted_mean:.2f}, returns_discounted_std: {returns_discounted_std:.2f}"
        )
        writer.add_scalar("Eval/train", returns_mean, agent.steps_interact)
        writer.add_scalar("Eval/train_discount", returns_discounted_mean, agent.steps_interact)
        for config_eval in configs_eval:
            env_generator = lambda: get_new_env(args, **config_eval)
            returns_mean, returns_std, returns_discounted_mean, returns_discounted_std = evaluate_agent(
                env_generator, agent, num_episodes=20, type_env="minigrid"
            )
            diff = np.mean(config_eval["lava_density_range"])
            print(
                f"Eval/{diff:g} x{20} @ step {agent.steps_interact:d} - returns_mean: {returns_mean:.2f}, returns_std: {returns_std:.2f}, returns_discounted_mean: {returns_discounted_mean:.2f}, returns_discounted_std: {returns_discounted_std:.2f}"
            )
            writer.add_scalar(f"Eval/{diff:g}", returns_mean, agent.steps_interact)
            writer.add_scalar(f"Eval/discount_{diff:g}", returns_discounted_mean, agent.steps_interact)
        pointer_milestone += 1
    if not (agent.steps_interact <= args.steps_max and episode_elapsed <= args.episodes_max and agent.steps_interact <= args.steps_stop):
        break
    while not done and agent.steps_interact <= args.steps_max:
        action = agent.decide(obs_curr, env=env, writer=writer, random_walk=bool(args.random_walk_leap))
        obs_next, reward, done, info = env.step(action)
        real_done = done and not info["overtime"]
        steps_episode += 1
        agent.step(obs_curr, action, reward, obs_next, real_done, writer=writer)
        return_cum += reward
        return_cum_discount += reward * args.gamma**env.step_count
        obs_curr = obs_next
    if done:
        agent.on_episode_end()
        time_episode_end = time.time()
        writer.add_scalar("Experience/return", return_cum, agent.steps_interact)
        writer.add_scalar("Experience/return_discount", return_cum_discount, agent.steps_interact)
        writer.add_scalar("Experience/dist2init", info["dist2init"], agent.steps_interact)
        writer.add_scalar("Experience/dist2goal", info["dist2goal"], agent.steps_interact)
        writer.add_scalar("Experience/dist2init_x", np.abs(info["agent_pos"][0] - info["agent_pos_init"][0]), agent.steps_interact)
        writer.add_scalar("Experience/overtime", float(info["overtime"]), agent.steps_interact)
        writer.add_scalar("Experience/episodes", episode_elapsed, agent.steps_interact)

        str_info += (
            f"seed: {args.seed}, steps_interact: {agent.steps_interact}, episode: {episode_elapsed}, "
            f"return: {return_cum: g}, return_discount: {return_cum_discount: g}, "
            f"steps_episode: {steps_episode}"
        )
        sps_averaged = agent.steps_interact / (time_episode_end - time_start)
        writer.add_scalar("Other/sps", sps_averaged, agent.steps_interact)
        eta = str(datetime.timedelta(seconds=int((args.steps_stop - agent.steps_interact) / sps_averaged)))
        str_info += ", sps_avg: %.2f, eta: %s" % (sps_averaged, eta)
        print(str_info)
        writer.add_text("Text/info_train", str_info, agent.steps_interact)

        return_cum, return_cum_discount, steps_episode, time_episode_start, str_info = (0, 0, 0, time.time(), "")
        episode_elapsed += 1

time_end = time.time()
time_duration = time_end - time_start
print("total time elapsed: %s" % str(datetime.timedelta(seconds=time_duration)))
