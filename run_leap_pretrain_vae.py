import time, datetime, numpy as np, os, pickle
from gym.envs.registration import register as gym_register

gym_register(id="RandDistShift-v2", entry_point="RandDistShift:RandDistShift2", reward_threshold=0.95)

from baselines import create_RW_agent

from tensorboardX import SummaryWriter
from runtime import generate_exptag, get_set_seed, get_new_env, config_parser, save_code_snapshot
import torch

from utils import process_batch, visualize_generation_minigrid2

from models import CVAE_MiniGrid_Separate2
from models import Encoder_MiniGrid_Separate, Decoder_MiniGrid_Separate
from utils import get_cpprb_env_dict, minigridobs2tensor
from HER import HindsightReplayBuffer

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

if args.num_envs_train > 0:
    envs_train = []
    for idx_env in range(args.num_envs_train):
        env = get_new_env(args, **config_train)
        env.reset()
        env.generate_oracle()
        envs_train.append(env)

    def generator_env_train():
        idx_env = np.random.randint(args.num_envs_train)
        return envs_train[idx_env]

else:

    def generator_env_train():
        env_train = get_new_env(args, **config_train)
        return env_train


args.method = "leap"
env = get_new_env(args, **config_train)
args = generate_exptag(args, additional="")
args.seed = get_set_seed(args.seed, env)

print(args)

agent = create_RW_agent(args, env)

################################################################
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_categoricals, num_categories = 6, 2
depth, width = 2, 256
atoms = 4
beta = 0.00025
debug = True
prioritized_cvae = True
freq_visualize_generation = 10000
eps_adam = 1.5e-4  # 1e-8  #
size_batch_cvae = args.size_batch  # 512  #
onehot_state = False
activation = torch.nn.ReLU
additional_goals = 4
interval_beta = 5000
unique_goals = False
local_comments = f"beta_interval{interval_beta:g}"

if onehot_state:
    local_comments += "_onehot"
else:
    local_comments += "_compact"
local_comments += "_unlimited_CVAE_buffer"
if prioritized_cvae:
    local_comments += "_prior"
else:
    local_comments += "_noprior"
if eps_adam != 1.5e-4:
    local_comments += f"_eps{eps_adam}"

if size_batch_cvae != args.size_batch:
    local_comments += f"_bs_cvae{size_batch_cvae:d}"

if unique_goals:
    local_comments += "_unique_goals"

while len(local_comments) and local_comments[0] == "_":
    local_comments = local_comments[1:]
while len(local_comments) and local_comments[-1] == "_":
    local_comments = local_comments[:-1]

env_dict = get_cpprb_env_dict(env)
hrb = HindsightReplayBuffer(
    additional_goals * args.size_buffer,
    env_dict,
    max_episode_len=env.unwrapped.max_steps,
    reward_func=None,
    prioritized=prioritized_cvae,
    strategy=args.hindsight_strategy,
    additional_goals=additional_goals,
    num_goals_per_transition=1,
    unique_goals=unique_goals,
)

layout_extractor = Encoder_MiniGrid_Separate()
decoder = Decoder_MiniGrid_Separate()

sample_layout, sample_mask_agent = layout_extractor(minigridobs2tensor(env.reset()))
cvae = CVAE_MiniGrid_Separate2(
    layout_extractor,
    decoder,
    minigridobs2tensor(env.reset()),
    num_categoricals=num_categoricals,
    num_categories=num_categories,
    beta=beta,
    activation=activation,
    interval_beta=interval_beta,
)


cvae.to(DEVICE)
params_cvae = cvae.parameters()
optimizer_cvae = torch.optim.Adam(params_cvae, lr=args.lr, eps=eps_adam)

################################################################

milestones_evaluation = []
step_milestone, pointer_milestone = 0, 0
while step_milestone <= args.steps_stop:
    milestones_evaluation.append(step_milestone)
    step_milestone += args.freq_eval

if args.uniform_init:
    path_tf_events = f"tb_records/{env.spec.id}/{args.size_world}x{args.size_world}/leap/vae_discrete_pretrain/{args.comments}/{args.seed}"
else:
    path_tf_events = f"tb_records/{env.spec.id}/{args.size_world}x{args.size_world}/leap/vae_discrete_pretrain_non_uniform/{args.comments}/{args.seed}"
writer = SummaryWriter(path_tf_events)
save_code_snapshot(path_tf_events)


episode_elapsed, step_last_eval = 0, -freq_visualize_generation
time_start = time.time()
return_cum, return_cum_discount, steps_episode, time_episode_start, str_info = 0.0, 0.0, 0, time.time(), ""

while True:
    if args.randomized:
        env = generator_env_train()
    obs_curr, done = env.reset(), False
    obs_init = obs_curr
    if not (agent.steps_interact <= args.steps_max and episode_elapsed <= args.episodes_max and agent.steps_interact <= args.steps_stop):
        break
    while not done and agent.steps_interact <= args.steps_max:
        action = agent.decide(obs_curr, env=env, writer=writer, random_walk=args.random_walk)
        obs_next, reward, done, info = env.step(action)
        real_done = done and not info["overtime"]
        ################################################
        if agent.steps_interact - step_last_eval >= freq_visualize_generation and not real_done:
            idx_config = np.random.choice(range(len(configs_eval)))
            config_eval = configs_eval[idx_config]
            env_debug = get_new_env(args, **config_eval)
            obs_cond = env_debug.reset()
            visualize_generation_minigrid2(cvae, obs_cond, env, writer, step_record=agent.steps_interact)
            step_last_eval += freq_visualize_generation
        sample = {"obs": obs_curr, "act": action, "rew": reward, "next_obs": obs_next, "done": real_done}
        hrb.add(**sample)
        # and agent.steps_interact >= agent.time_learning_starts
        if hrb.get_stored_size() > size_batch_cvae and agent.steps_interact % 4 == 0:
            batch = hrb.sample(size_batch_cvae)
            batch_processed = process_batch(batch, prioritized=prioritized_cvae, with_targ=True, obs2tensor=minigridobs2tensor, device=DEVICE, aux=False)
            cvae.train()
            (
                loss_overall,
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
            ) = cvae.compute_loss(batch_processed, debug=debug and agent.steps_interact % 100 == 0)
            if prioritized_cvae:
                weights_rb, idxes_rb = batch_processed[-2], batch["indexes"]
                loss_overall_weighted = (loss_overall * weights_rb.detach().squeeze()).mean()
            else:
                loss_overall_weighted = loss_overall.mean()
            optimizer_cvae.zero_grad(set_to_none=True)
            loss_overall_weighted.backward()
            torch.nn.utils.clip_grad_value_(params_cvae, 1.0)
            optimizer_cvae.step()
            with torch.no_grad():
                if prioritized_cvae:
                    loss_entropy_weighted = (loss_entropy * weights_rb.detach().squeeze()).mean()
                    loss_recon_weighted = (loss_recon * weights_rb.detach().squeeze()).mean()
                else:
                    loss_entropy_weighted = loss_entropy.mean()
                    loss_recon_weighted = loss_recon.mean()
                if prioritized_cvae:
                    hrb.update_priorities(idxes_rb, loss_overall.detach().cpu().numpy().squeeze())

            writer.add_scalar(f"Loss/recon", loss_recon_weighted.item(), agent.steps_interact)
            writer.add_scalar(f"Loss/entropy", loss_entropy_weighted.item(), agent.steps_interact)
            writer.add_scalar(f"Loss/overall", loss_overall_weighted.item(), agent.steps_interact)
            if debug and agent.steps_interact % 100 == 0:
                writer.add_scalar(f"Dist/L1", dist_L1_mean.item(), agent.steps_interact)
                writer.add_scalar(f"Dist/L1_nontrivial", dist_L1_nontrivial.item(), agent.steps_interact)
                writer.add_scalar(f"Dist/L1_trivial", dist_L1_trivial.item(), agent.steps_interact)
                writer.add_scalar(f"Dist/ratio_imperfect_recon", 1 - ratio_perfect_recon.item(), agent.steps_interact)
                writer.add_scalar(f"Dist/ratio_unaligned", 1 - ratio_aligned.item(), agent.steps_interact)
        ####################################
        steps_episode += 1
        agent.step(obs_curr, action, reward, obs_next, done and not info["overtime"], writer=writer)
        return_cum += reward
        return_cum_discount += reward * args.gamma**env.step_count
        obs_curr = obs_next
    if done:
        agent.on_episode_end()
        hrb.on_episode_end()
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
        duration_episode = time_episode_end - time_episode_start
        if duration_episode:
            sps_episode = steps_episode / duration_episode
            writer.add_scalar("Other/sps", sps_episode, agent.steps_interact)
            eta = str(datetime.timedelta(seconds=int((args.steps_stop - agent.steps_interact) / sps_episode)))
            str_info += ", sps_episode: %.2f, eta: %s" % (sps_episode, eta)
        print(str_info)
        writer.add_text("Text/info_train", str_info, agent.steps_interact)

        return_cum, return_cum_discount, steps_episode, time_episode_start, str_info = (0, 0, 0, time.time(), "")
        episode_elapsed += 1

time_end = time.time()
time_duration = time_end - time_start
print("total time elapsed: %s" % str(datetime.timedelta(seconds=time_duration)))
torch.save(
    {
        "steps_interact": agent.steps_interact,
        "model_state_dict": cvae.state_dict(),
        "num_categoricals": num_categoricals,
        "num_categories": num_categories,
    },
    os.path.join(path_tf_events, "cvae_leap.pt"),
)
if args.num_envs_train > 0:
    with open(os.path.join(path_tf_events, "envs.pkl"), "wb") as file:
        pickle.dump(envs_train, file)
