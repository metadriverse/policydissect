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

from policydissect.legged_gym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym
from isaacgym import gymapi
from policydissect.legged_gym.envs import *
from policydissect.legged_gym.utils import get_args, export_policy_as_jit, task_registry, Logger
from policydissect.legged_gym.policy_utils import ppo_inference_torch, control_neuron_activation
import numpy as np
import torch
from policydissect.legged_gym.pid import PIDController


def play(args, map, activation_func="elu", model_name=None):
    target_angular_v = 1.0
    pid_controller = PIDController(k_p=0.5, k_i=0.01, k_d=0.05)
    # import pygame module in this program
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
    env_cfg.no_obstacle = True
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

        obs[..., 12] = 0.
        obs[..., 10] = 1.
        obs[..., 11] = 0.
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
        if dones[0]:
            command = "Stop"


if __name__ == '__main__':
    anymal_tanh = {"Forward": {0: [(56, 15)]}}
    # (479, 30) (47, 32)
    args = get_args()
    args.num_envs = 1

    args.task = "anymal_c_flat"
    activation = "tanh"
    play(args, activation_func=activation, map=anymal_tanh, model_name="anymal")
