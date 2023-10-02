import torch.multiprocessing as multiprocessing
from utils import *
from runtime import generate_exptag, get_set_seed, get_new_env, config_parser
import utils_mp

if __name__ == "__main__":
    parser = config_parser(mp=True)
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

    env = get_new_env(args, **config_train)
    args = generate_exptag(args, additional="")
    args.seed = get_set_seed(args.seed, env)

    print(args)

    # MAIN
    multiprocessing.set_start_method("spawn")
    utils_mp.run_multiprocess(args, config_train, configs_eval)
