import argparse
import os
import os.path
import pickle

import numpy as np
import pybullet
from gym.spaces import Box

from policydissect.quadrupedal.torchrl.env.base_wrapper import BaseWrapper
from policydissect.quadrupedal.vision4leg.get_env import get_env
from policydissect.utils.legged_config import hrl_param
from policydissect.utils.legged_utils import seed_env
from policydissect.utils.policy import ppo_inference_torch
from policydissect.weights import weights_path


class HRLWrapper(BaseWrapper):
    REPEAT = None
    LEGGED_MAP = {"Turn Left": {3: [(239, 85)]},
                  "Turn Right": {3: [(239, -75)]},
                  "Stop": {1: [(76, -70)]}}

    @classmethod
    def set_repeat(cls, repeat):
        print("Set action repeat: {}".format(repeat))
        cls.REPEAT = repeat

    def __init__(self, env, repeat):
        super(HRLWrapper, self).__init__(env)
        assert repeat is not None, "If hrl = None then do not use this wrapper"
        self.REPEAT = repeat
        with open(os.path.join(weights_path, "quadrupedal_obs_normalizer.pkl"), 'rb') as f:
            env._obs_normalizer = pickle.load(f)
        self.policy_weights = np.load(os.path.join(weights_path, "quadrupedal.npz"))
        self.action_space = Box(low=-1., high=1., shape=(1,))
        self._actions = ["Forward", "Turn Left", "Turn Right", "Stop"]
        self.last_o = None

    def need_image(self):
        self.env.env.env.env.env.env._gym_env.get_image_interval = 1

    def no_image(self):
        self.env.env.env.env.env.env._gym_env.get_image_interval = 10000000000

    def reset(self, **kwargs):
        self.need_image()
        ret = super(HRLWrapper, self).reset(**kwargs)
        self.last_o = ret[:93]
        return ret

    def step(self, action):
        if -1 <= action < -0.5:
            action = 0
        elif -0.5 <= action < 0.:
            action = 1
        elif 0. <= action < 0.5:
            action = 2
        elif 0.5 <= action <= 1.:
            action = 3
        else:
            raise ValueError("out of bound")
        command = self._actions[action]
        for i in range(self.REPEAT):
            action = ppo_inference_torch(self.policy_weights, self.last_o, self.LEGGED_MAP, command)
            if i == self.REPEAT - 1:
                self.need_image()
            else:
                self.no_image()
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
    params["env"]["env_build"]["terrain_type"] = "plane"
    # params["env"]["env_build"]["terrain_type"] = "random_blocks_sparse_and_heightfield"

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
    env.REPEAT = 100
    env.reset()
    for command in [1, 0, 2, 3, 1, 0, 2, 3, ]:
        command = (command - 2) * 0.5
        o, r, d, i = env.step(command)
        if d:
            env.reset()
