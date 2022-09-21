import os
import pickle
import time
import isaacgym
from isaacgym.gymapi import *
from isaacgym.gymtorch import *

import numpy as np
from isaacgym import gymapi
from isaacgym.torch_utils import quat_apply

from policydissect import PACKAGE_DIR
from policydissect.legged_gym.pid import ActivationPID, PIDController
from policydissect.legged_gym.policy_utils import ppo_inference_torch
from policydissect.legged_gym.utils import task_registry, Logger
import torch


def follow_command(
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
    policy_weights = np.load(os.path.join(PACKAGE_DIR, "weights", "{}_{}.npz".format(name, activation_func)))
    logger = Logger(env.dt)

    # run
    time.sleep(3)
    while True:
        for target_heading in target_heading_list:
            for i in range(command_last_time):
                forward = quat_apply(self.base_quat, self.forward_vec)
                heading = torch.atan2(forward[:, 1], forward[:, 0])
                error = heading - target_heading
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

                print("Explicit Command in Observation: {}".format(obs[..., 10:13]))
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


def play(args, map, activation_func="elu", model_name=None):
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
    policy_weights = np.load(os.path.join(PACKAGE_DIR, "weights", "{}_{}.npz".format(name, activation_func)))
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

        obs[..., 10] = 0.1  # Default Stop
        actions, _ = ppo_inference_torch(
            policy_weights, obs.clone().cpu().numpy(), map, command, activation=activation_func, deterministic=True
        )
        actions = torch.unsqueeze(torch.from_numpy(actions.astype(np.float32)), dim=0)
        obs, _, rews, dones, infos, = env.step(actions)
        x, y, z = env.base_pos[0]
        env.set_camera((x - 3, y, 2), (x, y, z))
        counter -= 1
        if dones[0]:
            command = "Stop"
