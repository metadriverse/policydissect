from legged_gym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, export_policy_as_jit, task_registry, Logger
from legged_gym.policy_utils import ppo_inference_torch, control_neuron_activation
import numpy as np
import torch
import pickle
import time
from isaacgym import *
from legged_gym import *

import numpy as np
from legged_gym.policy_utils import ppo_inference_torch
from legged_gym.utils import get_args, task_registry

import os
import librosa
import matplotlib.pyplot as plt

import pickle
import numpy as np


def cal_relation(data_1, data_2):
    # if np.linalg.norm(data_1) != 0:
    #     data_1 = data_1 / np.linalg.norm(data_1)
    # if np.linalg.norm(data_2) != 0:
    #     data_2 = data_2 / np.linalg.norm(data_2)
    # assert len(data_1) == len(data_2), "d_1:{}, d_2:{}".format(len(data_1), len(data_2))
    # error = 0
    # for i in range(len(data_1)):
    #     error += abs(data_1[i] - data_2[i])
    return np.linalg.norm(data_1 - data_2)


def get_most_relevant_neuron(neurons_activation_fft, epi_target_dims_fft, target_dim_name="obs"):
    target_ret = {}
    neuron_ret = {}

    for k, target_dim, in enumerate(epi_target_dims_fft):
        target_error = []
        print("============ process {} dim: {} ============".format(target_dim_name, k))
        for layer in range(len(neurons_activation_fft)):
            if layer not in neuron_ret:
                neuron_ret[layer] = {}
            for neuron_index in range(len(neurons_activation_fft[layer])):
                if neuron_index not in neuron_ret[layer]:
                    neuron_ret[layer][neuron_index] = []
                neuron_fft = neurons_activation_fft[layer][neuron_index]["fft_amplitude"]
                target_dim_fft = target_dim["fft_amplitude"]

                neuron_phase = neurons_activation_fft[layer][neuron_index]["fft_phase"]
                target_phase = target_dim["fft_phase"]
                phase_diff = neuron_phase - target_phase
                base_freq = np.argmax(np.sum(neuron_fft - target_dim_fft, axis=1))

                relation_coefficient = -2 * abs(np.mean(phase_diff[base_freq]) / np.pi) + 1

                error_freq = cal_relation(neuron_fft, target_dim_fft)
                # error = error_norm*error_freq
                target_error.append({"neuron": {"layer": layer, "neuron_index": neuron_index},
                                     "error": {"freq_diff": error_freq, "base_phase_diff": relation_coefficient},
                                     })
                neuron_ret[layer][neuron_index].append(
                    {"{}_dim".format(target_dim_name): k,
                     "error": {"freq_diff": error_freq, "base_phase_diff": relation_coefficient}})
        target_error.sort(key=lambda i: i["error"]["freq_diff"])
        target_ret[k] = target_error

    for layer in neuron_ret.keys():
        for neuron_index in neuron_ret[layer].keys():
            neuron_ret[layer][neuron_index].sort(key=lambda i: i["error"]["freq_diff"])
    key = "{}_analysis".format(target_dim_name)
    return {key: target_ret, "neuron_analysis": neuron_ret}


def draw_origin_and_fft(data, label=None, save_figure=False, n_fft=16, for_neuron=True):
    plt.clf()
    plt.cla()
    assert label is not None, "assign a label for figure"

    y = data
    ywf = librosa.stft(y, n_fft=n_fft)
    magnitude = np.abs(ywf)
    phase = np.angle(ywf)
    plot_y = magnitude

    if save_figure:
        plt.subplot(3, 1, 1)
        plt.plot(data, label="Activation of Unit: {}".format(label) if for_neuron else "Transition of {}".format(label))
        plt.legend()
        plt.subplot(3, 1, 2)
        plt.imshow(plot_y, aspect="auto", origin="lower",
                   interpolation='none')
        plt.legend("STFT, {}".format(label))
        plt.subplot(3, 1, 3)
        plt.imshow(phase, aspect="auto", origin="lower",
                   interpolation='none')
        plt.legend("STFT Phase, {}".format(label))
        plt.savefig("./dissection/{}.png".format(label))
    return plot_y, phase


def axis_shift(epi_activation, label="after_tanh"):
    acivation_per_step = []
    for layers_per_step in epi_activation:
        layers = []
        for layer in [x[label][0] for x in layers_per_step]:
            if len(layer) != 512:
                layers.append(np.concatenate([layer, np.zeros([512 - len(layer)])]))
            else:
                layers.append(layer)
        concat_ret = np.array(layers)
        acivation_per_step.append(concat_ret)
    acivation_per_step = np.array(acivation_per_step)
    acivation_per_step = np.moveaxis(acivation_per_step, 0, -1)
    return acivation_per_step


def analyze_neuron(epi_activation, save_figure=False, n_fft=16, specific_neuron=None):
    activation_after_tanh = axis_shift(epi_activation)
    # activation_before_tanh = axis_shift(epi_activation, label="before_tanh")
    neurons_fft = []
    for layer in range(len(activation_after_tanh)):
        layer_fft = []
        for neuron in range(len(activation_after_tanh[layer])):
            if specific_neuron is None:
                fft_ret, phase = draw_origin_and_fft(activation_after_tanh[layer][neuron],
                                                     label="neuron_{}_{}".format(layer, neuron),
                                                     save_figure=save_figure,
                                                     for_neuron=True,
                                                     n_fft=n_fft)
                strength_dist = activation_after_tanh[layer][neuron]
                # positive_strength = np.quantile(strength_dist, strength_quantile)
                # negative_strength = np.quantile(strength_dist, 1 - strength_quantile)
                layer_fft.append({"fft_amplitude": fft_ret, "fft_phase": phase,
                                  # "strength": {"positive": positive_strength, "negative": negative_strength}
                                  })
            elif specific_neuron is not None:
                assert isinstance(specific_neuron, list), "Use list [(layer, neuron index), (),...]"
                if (layer, neuron) in specific_neuron:
                    fft_ret, phase = draw_origin_and_fft(activation_after_tanh[layer][neuron],
                                                         label="neuron_{}_{}".format(layer, neuron),
                                                         save_figure=save_figure,
                                                         for_neuron=True,
                                                         n_fft=n_fft)
                    strength_dist = activation_after_tanh[layer][neuron]
                    # positive_strength = np.quantile(strength_dist, strength_quantile)
                    # negative_strength = np.quantile(strength_dist, 1 - strength_quantile)
                    layer_fft.append({"fft_amplitude": fft_ret, "fft_phase": phase,
                                      # "strength": {"positive": positive_strength, "negative": negative_strength}
                                      })
                else:
                    layer_fft.append({"fft_amplitude": np.inf, "fft_phase": np.inf,
                                      # "strength": {"positive": positive_strength, "negative": negative_strength}
                                      })

        neurons_fft.append(layer_fft)
    return neurons_fft, activation_after_tanh


def analyze_observation(epi_observation, save_figure=False, n_fft=16, specific_obs=None):
    obs_per_step = np.array(epi_observation)
    per_obs_dim = np.moveaxis(obs_per_step, 0, -1)
    obs_fft = []
    for dim in range(len(per_obs_dim)):
        if specific_obs is None:
            fft_ret, phase = draw_origin_and_fft(per_obs_dim[dim],
                                                 label="obs_dim_{}".format(dim),
                                                 save_figure=save_figure,
                                                 for_neuron=False,
                                                 n_fft=n_fft)
            obs_fft.append({"fft_amplitude": fft_ret, "fft_phase": phase})
        elif specific_obs is not None:
            if dim in specific_obs:
                fft_ret, phase = draw_origin_and_fft(per_obs_dim[dim],
                                                     label="obs_dim_{}".format(dim),
                                                     save_figure=save_figure,
                                                     for_neuron=False,
                                                     n_fft=n_fft)
                obs_fft.append({"fft_amplitude": fft_ret, "fft_phase": phase})
            else:
                # print("discard obs dim: {}".format(dim))
                # print("This may cause reindex error !!! "
                #       "since the processed obs index will change after discarding useless obs dim!!!")
                pass
    return obs_fft, per_obs_dim


def analyze_actions(epi_action, save_figure=False, n_fft=16):
    obs_per_step = np.array(epi_action)
    per_action_dim = np.moveaxis(obs_per_step, 0, -1)
    action_fft = []
    for dim in range(len(per_action_dim)):
        fft_ret, phase = draw_origin_and_fft(per_action_dim[dim],
                                             label="action_dim_{}".format(dim),
                                             save_figure=save_figure,
                                             for_neuron=False,
                                             n_fft=n_fft)
        action_fft.append({"fft_amplitude": fft_ret, "fft_phase": phase})
    return action_fft, per_action_dim


def do_policy_dissection(collect_episodes, specific_neuron=None, specific_obs=None, obs_neuron_pair=None):
    n_fft = 32
    # assert not os.path.exists("dissection"), "please save previous result"
    # os.makedirs("dissection")
    ckpt_ret = {}
    for k, epi_data in enumerate(collect_episodes):
        print("===== Dissect episode {} =====".format(k))
        epi_activation = epi_data["neuron_activation"]
        observations = epi_data["observations"]

        neurons_fft, origin_neuron = analyze_neuron(epi_activation, n_fft=n_fft,
                                                    specific_neuron=specific_neuron)

        obs_fft, origin_obs = analyze_observation(observations, n_fft=n_fft,
                                                  specific_obs=specific_obs)

        ret_obs = get_most_relevant_neuron(target_dim_name="obs",
                                           neurons_activation_fft=neurons_fft,
                                           epi_target_dims_fft=obs_fft)["obs_analysis"]
        this_epi_frequency_error = ret_obs
        this_epi_frequency_error["seed"] = k
        ckpt_ret[k] = this_epi_frequency_error
    return ckpt_ret


def make_env(task_name="cassie"):
    args = get_args()
    args.num_envs = 1
    args.task = task_name
    args.headless = True
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
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
    policy_func = ppo_inference_torch
    # policy_func = ppo_inference
    seed_num = 10
    start_time = time.time()

    path = "./scripts/anymal_tanh.npz"
    activation_func = "tanh"
    task_name = "anymal_c_flat"
    # task_name = "cassie"
    command = None

    env = make_env(task_name=task_name)
    weights = np.load(path)
    print("===== Do Policy Dissection for {} ckpt =====".format(path))
    collected_episodes = []
    for seed in range(seed_num):
        o, _ = env.reset()
        episode_activation_values = []
        episode_observations = [o.cpu().numpy()[0]]
        current_step = 0
        total_r = 0

        while True:
            if command is not None:
                o[..., 9:12] = torch.Tensor(command)
            action, activation = policy_func(weights, o.clone().cpu().numpy(), {}, "", activation=activation_func)
            o, _, r, d, i = env.step(torch.unsqueeze(torch.from_numpy(action.astype(np.float32)), dim=0))
            episode_activation_values.append(activation)
            current_step += 1
            total_r += r
            if d:
                collected_episodes.append(dict(neuron_activation=episode_activation_values,
                                               observations=episode_observations))
                print("Finish seed: {}, reward: {}".format(seed, total_r))
                break
            episode_observations.append(o.cpu().numpy()[0])
    self = env
    self.gym.destroy_sim(self.sim)
    if self.viewer is not None:
        self.gym.destroy_viewer(self.viewer)
    with open("collect_episodes.pkl", "wb+") as epi_data:
        pickle.dump(collected_episodes, epi_data)
    pd_ret = do_policy_dissection(collected_episodes)
    with open("{}.pkl".format("policy_dissection_ret"), "wb+") as file:
        pickle.dump(pd_ret, file)
