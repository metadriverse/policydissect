import random
from policydissect.quadrupedal.vision4leg.get_env import get_single_env


import numpy as np


def seed_env(env, seed):
    env.eval()
    env.seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def get_single_hrl_env(env_id, env_param):
    from policydissect.utils.legged_hrl_env import HRLWrapper
    env = get_single_env(env_id, env_param)
    return HRLWrapper(env)
