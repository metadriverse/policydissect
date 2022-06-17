import argparse
import os.path
import pickle

import keyboard
import numpy as np
import pybullet

from policydissect.quadrupedal.vision4leg.get_env import get_env
from policydissect.utils.legged_utils import params, seed_env
from policydissect.utils.policy import ppo_inference_torch
from policydissect.weights import weights_path

legged_robot_command = "Forward"


def update_render(env):
    pos = env.env.env.env.env.env.env._gym_env._robot.GetBasePosition()
    text_pos = [pos[0] - 0.5, pos[1], pos[2] + 0.5]
    pybullet.addUserDebugText(legged_robot_command, text_pos, textSize=4, textColorRGB=[1, 0, 0],
                              replaceItemUniqueId=True)


def legged_control(event):
    global legged_robot_command
    if event.event_type == "down" and event.name == "i":
        legged_robot_command = "Forward"
    elif event.event_type == "down" and event.name == "l":
        legged_robot_command = "Turn Right"
    elif event.event_type == "down" and event.name == "j":
        legged_robot_command = "Turn Left"
    elif event.event_type == "down" and event.name == "k":
        legged_robot_command = "Stop"
    elif event.event_type == "down" and event.name == "r":
        legged_robot_command = "Reset"


LEGGED_MAP = {"Turn Left": {3: [(239, 85)]},
              "Turn Right": {3: [(239, -75)]},
              "Stop": {1: [(76, -70)]}}

if __name__ == "__main__":

    keyboard.hook(legged_control)
    parser = argparse.ArgumentParser()
    parser.add_argument('--hard', action="store_true")
    args = parser.parse_args()
    params["env"]["env_build"]["enable_rendering"] = True
    if args.hard:
        params["env"]["env_build"]["terrain_type"] = "random_blocks_sparse_and_heightfield"
        # params["env"]["env_build"]["terrain_type"] = "random_blocks_sparse_thin_wide"
    else:
        params["env"]["env_build"]["terrain_type"] = "plane"

    env = get_env(
        params['env_name'],
        params['env'])

    pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_WIREFRAME, 0)
    pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
    pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
    pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
    pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, 0)
    pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_SHADOWS, 0)

    seed = 2
    seed_env(env, seed)
    pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, 0, lightPosition=(0, -50, 2))
    with open(os.path.join(weights_path, "quadrupedal_obs_normalizer.pkl"), 'rb') as f:
        env._obs_normalizer = pickle.load(f)
    policy_weights = np.load(os.path.join(weights_path, "quadrupedal.npz"))

    o = env.reset()
    while True:
        if legged_robot_command == "Reset":
            o = env.reset()
        action = ppo_inference_torch(policy_weights, o, LEGGED_MAP, legged_robot_command)
        o, r, d, i = env.step(action)
        update_render(env)
        if d:
            seed_env(env, seed)
            env.reset()
