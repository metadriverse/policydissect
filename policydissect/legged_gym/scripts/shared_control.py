# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym
from isaacgym import gymapi
from legged_gym.envs import *
from legged_gym.utils import get_args, export_policy_as_jit, task_registry, Logger
from legged_gym.policy_utils import ppo_inference_torch, control_neuron_activation
import numpy as np
import torch


def play(args, map, activation_func="elu", model_name=None):
    # import pygame module in this program
    import pygame

    # activate the pygame library
    # initiate pygame and give permission
    # to use pygame's functionality.
    pygame.init()

    # define the RGB value for white,
    #  green, blue colour .
    white = (255, 255, 255)
    green = (0, 255, 0)
    blue = (0, 0, 128)

    # assigning values to X and Y variable
    X = 200
    Y = 1000

    # create the display surface object
    # of specific dimension..e(X, Y).
    display_surface = pygame.display.set_mode((X, Y))

    # set the pygame window name
    pygame.display.set_caption('Show Text')

    # create a font object.
    # 1st parameter is the font file
    # which is present in pygame.
    # 2nd parameter is size of the font
    font = pygame.font.Font('freesansbold.ttf', 32)

    assert activation_func == "elu" or activation_func == "tanh", "only support elu or tanh"
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 50)
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.no_obstacle = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.terrain.mesh_type = "plane"
    train_cfg.runner.num_steps_per_env = 1

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    env.max_episode_length = 10000

    env.gym.subscribe_viewer_keyboard_event(env.viewer, gymapi.KEY_W, "Forward")
    env.gym.subscribe_viewer_keyboard_event(env.viewer, gymapi.KEY_A, "Left")
    env.gym.subscribe_viewer_keyboard_event(env.viewer, gymapi.KEY_S, "Stop")
    env.gym.subscribe_viewer_keyboard_event(env.viewer, gymapi.KEY_C, "Crouch")
    env.gym.subscribe_viewer_keyboard_event(env.viewer, gymapi.KEY_X, "Tiptoe")
    env.gym.subscribe_viewer_keyboard_event(env.viewer, gymapi.KEY_SPACE, "Back Flip")
    env.gym.subscribe_viewer_keyboard_event(env.viewer, gymapi.KEY_Q, "Jump")
    env.gym.subscribe_viewer_keyboard_event(env.viewer, gymapi.KEY_R, "Reset")

    env.gym.subscribe_viewer_keyboard_event(env.viewer, gymapi.KEY_D, "Right")

    self = env
    name = model_name or ("anymal" if "anymal" in args.task else "cassie")
    policy_weights = np.load("{}_{}.npz".format(name, activation_func))
    command = "Stop"
    counter = 0

    # env.gym.attach_camera_to_body(camera_handle, env.envs[0], env.gym.find_actor_rigid_body_handleenv.actor_handles[0])
    for i in range(10 * int(env.max_episode_length)):
        for evt in self.gym.query_viewer_action_events(self.viewer):
            if evt.value > 0 and evt.action in map.keys():
                command = evt.action
                if command == "Back Flip":
                    counter = 22
                elif command == "Jump":
                    counter = 40
            if evt.value > 0 and evt.action == "Reset":
                obs, _ = env.reset()

        if command == "Back Flip" and counter == 0:
            command = "Stop"

        if command == "Jump" and counter == 0:
            command = "Stop"

        obs[..., 10] = 0.1
        # obs[..., -121:] = 0.
        actions, _ = ppo_inference_torch(
            policy_weights, obs.clone().cpu().numpy(), map, command, activation=activation_func, deterministic=True
        )
        actions = torch.unsqueeze(torch.from_numpy(actions.astype(np.float32)), dim=0)
        obs, _, rews, dones, infos, = env.step(actions)
        x, y, z = env.base_pos[0]
        env.set_camera((x - 3, y, 2), (x, y, z))
        # env.render()
        counter -= 1
        display_surface.fill(white)

        # copying the text surface object
        # to the display surface object
        # at the center coordinate.
        # on which text is drawn on it.
        text = font.render(command, True, green, blue)

        # create a rectangular object for the
        # text surface object
        textRect = text.get_rect()

        # set the center of the rectangular object.
        textRect.center = (X // 2, Y // 2)
        display_surface.blit(text, textRect)
        pygame.display.update()
        if dones[0]:
            command = "Stop"


if __name__ == '__main__':
    cassie_elu = {"Right": {1: [(169, 10)]}, "Forward": {2: [(2, 5)]}, "z": {0: [(267, 7), (497, 10)]}}
    z_0_cassie_elu = {
        "Right": {
            1: [(169, 10)]
        },
        "Forward": {
            1: [(40, 28)],
            0: [(487, -10)]
        },
        "z": {
            0: [(394, 146), (170, 143), (487, -100)]
        }
    }
    anymal_elu = {"Forward": {0: [(143, -15)]}}

    forward_cassie = {
        # 0
        "Tiptoe": {
            0: [(261, 10), (212, 5), (395, 5), (260, 10), (30, -1)]
        },
        "Crouch": {
            0: [(261, -12), (260, 6)]
        },
        # 0,3
        "Back Flip": {
            0: [(47, 50), (479, 40), (261, 45)]
        },
        # 2
        "Left": {
            0: [(30, 10)]
        },
        "Right": {
            0: [(30, -9), (452, 20), (261, -9)]
        },
        # 1
        "Forward": {
            0: [(452, 20), (30, -1), (227, 8)]
        },
        # 21/15 hip flexion
        "Jump": {
            0: [(212, 25), (395, 25), (261, 10), (260, 5), (30, -15), (227, 12)]
        },
        "Stop": {}
    }
    # (479, 30) (47, 32)
    args = get_args()
    args.num_envs = 1

    # args.task = "anymal_c_rough"
    args.task = "cassie"
    activation = "elu"
    play(args, activation_func=activation, map=forward_cassie, model_name="forward_cassie")
