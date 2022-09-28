from isaacgym import gymapi
from policydissect.legged_gym.envs import *
from policydissect.legged_gym.utils import get_args
from policydissect.utils.isaacgym_utils import replay_cassie
from policydissect import PACKAGE_DIR
import os

if __name__ == "__main__":
    # replay the record
    args = get_args(
        [{
            "name": "--parkour",
            "action": "store_true",
            "default": False,
            "help": "Build a parkour environment"
        }]
    )
    args.num_envs = 1
    args.task = "cassie"
    activation = "elu"
    replay_cassie(args, os.path.join(PACKAGE_DIR, "scripts", "paper_demo.pkl"), parkour=True, force_seed=100)
