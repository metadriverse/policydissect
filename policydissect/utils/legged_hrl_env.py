import argparse
import os
import os.path
import pickle

import numpy as np
import pybullet
from gym.spaces import Discrete

from policydissect.quadrupedal.torchrl.env.base_wrapper import BaseWrapper
from policydissect.quadrupedal.vision4leg.get_env import get_env
from policydissect.utils.legged_utils import seed_env
from policydissect.utils.legged_config import hrl_param
from policydissect.utils.policy import ppo_inference_torch
from policydissect.weights import weights_path


class HRLWrapper(BaseWrapper):
    REPEAT = 20
    LEGGED_MAP = {"Turn Left": {3: [(239, 85)]},
                  "Turn Right": {3: [(239, -75)]},
                  "Stop": {1: [(76, -70)]}}

    def __init__(self, env):
        super(HRLWrapper, self).__init__(env)
        with open(os.path.join(weights_path, "quadrupedal_obs_normalizer.pkl"), 'rb') as f:
            env._obs_normalizer = pickle.load(f)
        self.policy_weights = np.load(os.path.join(weights_path, "quadrupedal.npz"))
        self.action_space = Discrete(4)
        self._actions = ["Forward", "Turn Left", "Turn Right", "Stop"]
        self.last_o = None

    def reset(self, **kwargs):
        ret = super(HRLWrapper, self).reset(**kwargs)
        self.last_o = ret[:93]
        return ret

    def step(self, action):
        command = self._actions[action]
        for _ in range(self.REPEAT):
            action = ppo_inference_torch(self.policy_weights, self.last_o, self.LEGGED_MAP, command)
            o, r, d, i = super(HRLWrapper, self).step(action)
            self.last_o = o[:93]
        return o, r, d, i


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--hard', action="store_true")
    parser.add_argument('--seed', default=0, type=int)
    args = parser.parse_args()
    params = hrl_param
    params["env"]["env_build"]["enable_rendering"] = True
    params["env"]["env_build"]["terrain_type"] = "random_blocks_sparse_and_heightfield"

    env = get_env(
        params['env_name'],
        params['env'])

    pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_WIREFRAME, 0)
    pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_KEYBOARD_SHORTCUTS, 0)
    pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
    pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
    pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
    pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, 0)
    pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_SHADOWS, 0)

    seed = args.seed
    seed_env(env, seed)
    env = HRLWrapper(env)
    env.reset()
    for command in [1, 0, 2, 3, 1, 0, 2, 3, ]:
        print(env._actions[command])
        o, r, d, i = env.step(command)
        if d:
            env.reset()
