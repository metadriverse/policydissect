import pickle
import time

import librosa
import matplotlib.pyplot as plt
import numpy
import numpy as np
import isaacgym
from policydissect.legged_gym.envs import *
from policydissect.policydissect import do_policy_dissection

from policydissect.legged_gym.policy_utils import ppo_inference_torch
from policydissect.legged_gym.utils import get_args, task_registry
import torch


def make_env(env_cfg, train_cfg, task_name="cassie"):
    args = get_args()
    args.num_envs = 1
    args.task = task_name
    args.headless = True
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 50)
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.terrain.mesh_type = "plane"
    train_cfg.runner.num_steps_per_env = 1

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    return env


if __name__ == "__main__":
    """
    This script dissect the anymal policy without command input. Though its training goal is moving forward, we can 
    still control it by finding angular velocity related neurons.
    """
    import random

    seed = 10
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    policy_func = ppo_inference_torch
    seed_num = 20
    start_time = time.time()

    path = "../policydissect/weights/anymal_forward_tanh.npz"
    activation_func = "tanh"
    task_name = "anymal_c_flat_forward"
    env_cfg, train_cfg = task_registry.get_cfgs(task_name)

    env = make_env(env_cfg=env_cfg, train_cfg=train_cfg, task_name=task_name)
    weights = np.load(path)
    print("===== Do Policy Dissection =====".format(path))
    collected_episodes = []
    for seed in range(seed_num):
        o, _ = env.reset()
        episode_activation_values = []
        episode_observations = [o.cpu().numpy()[0]]
        current_step = 0
        total_r = 0

        while True:
            action, activation = policy_func(
                weights, o.clone().cpu().numpy(), {}, "", activation=activation_func, deterministic=True
            )
            o, _, r, d, i = env.step(torch.unsqueeze(torch.from_numpy(action.astype(np.float32)), dim=0))
            episode_activation_values.append(activation)
            current_step += 1
            total_r += r
            if d:
                collected_episodes.append(
                    dict(neuron_activation=episode_activation_values, observations=episode_observations)
                )
                print("Finish seed: {}, reward: {}".format(seed, total_r))
                break
            episode_observations.append(o.cpu().numpy()[0])
    self = env
    self.gym.destroy_sim(self.sim)
    if self.viewer is not None:
        self.gym.destroy_viewer(self.viewer)

    pd_ret = do_policy_dissection(collected_episodes)
    with open("{}.pkl".format("anymal_ret"), "wb+") as file:
        pickle.dump(pd_ret, file)
