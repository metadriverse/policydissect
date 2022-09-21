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
import time

from legged_gym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym
from isaacgym import gymapi
from legged_gym.envs import *
from legged_gym.utils import get_args, export_policy_as_jit, task_registry, Logger
from legged_gym.policy_utils import ppo_inference_torch, control_neuron_activation
import numpy as np
import torch
from legged_gym.pid import ActivationPID, PIDController
import pickle
from isaacgym.torch_utils import quat_apply


def play(
    args,
    layer,
    index,
    activation_func="elu",
    model_name=None,
    target_heading_list=[0.5],
    command_last_time=50,
    trigger_neuron=True
):
    activation_pid_controller = ActivationPID(
        k_p=20,
        k_i=0.01,
        k_d=0.0,
        neuron_layer=layer,
        neuron_index=index,
    )
    pid_controller = PIDController(k_p=1.5, k_i=0.01, k_d=0.0)
    # import pygame module in this program
    assert activation_func == "elu" or activation_func == "tanh", "only support elu or tanh"
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 50)
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.no_obstacle = False
    env_cfg.commands.heading_command = True
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

    self = env
    name = model_name or ("anymal" if "anymal" in args.task else "cassie")
    policy_weights = np.load("{}_{}.npz".format(name, activation_func))
    logger = Logger(env.dt)

    # run
    time.sleep(3)
    while True:
        for target_heading in target_heading_list:
            for i in range(command_last_time):
                forward = quat_apply(self.base_quat, self.forward_vec)
                heading = torch.atan2(forward[:, 1], forward[:, 0])
                error = heading - target_heading
                print(error)
                obs[..., 10] = 1.  # only x velocity
                obs[..., 11] = 0.
                if trigger_neuron:
                    obs[..., 12] = 0.
                    activation_map = activation_pid_controller.get_updated_activation(
                        error.cpu().numpy(), command="Control"
                    )
                else:
                    obs[..., 12:13] = torch.tensor(pid_controller.get_result(error.cpu().numpy()))
                    activation_map = {}

                actions, _ = ppo_inference_torch(
                    policy_weights,
                    obs.clone().cpu().numpy(),
                    activation_map,
                    "Control" if trigger_neuron else "",
                    activation=activation_func,
                    deterministic=True
                )
                actions = torch.unsqueeze(torch.from_numpy(actions.astype(np.float32)), dim=0)
                obs, _, rews, dones, infos, = env.step(actions)
                x, y, z = env.base_pos[0]
                env.set_camera((x - 3, y, 2), (x, y, z))
                logger.log_states({
                    'command_heading': target_heading,
                    'base_heading': heading.cpu().numpy(),
                })
        logger.plot_states()
        with open("tracking_ret_{}.pkl".format("pd" if trigger_neuron else "goal"), "wb+") as file:
            pickle.dump(logger.state_log, file)


if __name__ == '__main__':
    # (479, 30) (47, 32)
    args = get_args()
    args.num_envs = 1

    args.task = "anymal_c_flat"
    activation = "tanh"
    play(
        args,
        activation_func=activation,
        layer=0,
        index=121,
        model_name="anymal",
        target_heading_list=[0, 0.4, 0.9, 1.2, 0.9, 0.4, 0., -0.4, -0.9, -1.2, -0.9, -0.4, 0.],
        command_last_time=100,
        trigger_neuron=True
    )
