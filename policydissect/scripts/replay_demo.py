from isaacgym import gymapi
from policydissect.legged_gym.envs import *
from policydissect.legged_gym.utils import get_args
from policydissect.utils.isaacgym_utils import replay_cassie
from policydissect import PACKAGE_DIR
import os

if __name__ == "__main__":
    # replay the record
    args = get_args()
    args.num_envs = 1
    args.task = "cassie"
    activation = "elu"
    replay_cassie(
        args, os.path.join(PACKAGE_DIR, "scripts", "paper_demo.pkl"), parkour=True, force_seed=100, frame_sus=0.03
    )
