import librosa
import matplotlib.pyplot as plt
import numpy as np


def cal_relation(data_1, data_2):
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
                # phase_diff = neuron_phase - target_phase
                # base_freq = np.argmax(
                #     np.sum(np.linalg.norm(neuron_fft - target_dim_fft, axis=1, keepdims=True), axis=1)
                # )
                #
                # # phase_diff = (phase_diff + np.pi) % (2 * np.pi) - np.pi
                # relation_coefficient = -2 * abs(np.mean(phase_diff[base_freq]) / np.pi) + 1

                # base_freq_neuron = np.argmax(np.sum(neuron_fft, axis=1))
                base_freq_obs = np.argmax(np.sum(target_dim_fft, axis=1))
                phase_diff = neuron_phase[base_freq_obs] - target_phase[base_freq_obs]

                phase_diff %= 2 * np.pi
                phase_diff -= 2 * np.pi * (phase_diff > np.pi)
                relation_coefficient = -2 * abs(np.mean(phase_diff) / np.pi) + 1

                error_freq = cal_relation(neuron_fft, target_dim_fft)
                # error = error_norm*error_freq
                target_error.append(
                    {
                        "neuron": {
                            "layer": layer,
                            "neuron_index": neuron_index
                        },
                        "error": {
                            "freq_diff": error_freq,
                            "correlation": relation_coefficient
                        },
                    }
                )
                neuron_ret[layer][neuron_index].append(
                    {
                        "{}_dim".format(target_dim_name): k,
                        "error": {
                            "freq_diff": error_freq,
                            "correlation": relation_coefficient
                        }
                    }
                )
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
        plt.imshow(plot_y, aspect="auto", origin="lower", interpolation='none')
        plt.legend("STFT, {}".format(label))
        plt.subplot(3, 1, 3)
        plt.imshow(phase, aspect="auto", origin="lower", interpolation='none')
        plt.legend("STFT Phase, {}".format(label))
        plt.savefig("./dissection/{}.png".format(label))
    return plot_y, phase


def axis_shift(epi_activation, label="after_tanh"):
    acivation_per_step = []
    for layers_per_step in epi_activation:
        layers = []
        unit_num_per_layer = [len(x[label][0]) for x in layers_per_step]
        for layer in [x[label][0] for x in layers_per_step]:
            # padding dim
            if len(layer) != max(unit_num_per_layer):
                layers.append(np.concatenate([layer, np.zeros([max(unit_num_per_layer) - len(layer)])]))
            else:
                layers.append(layer)
        concat_ret = np.array(layers)
        acivation_per_step.append(concat_ret)
    acivation_per_step = np.array(acivation_per_step)
    acivation_per_step = np.moveaxis(acivation_per_step, 0, -1)

    # Remove padding dims
    ret = []
    for k, unit_num in enumerate(unit_num_per_layer):
        ret.append(acivation_per_step[k][:unit_num][:])
    return np.array(ret)


def analyze_neuron(epi_activation, save_figure=False, n_fft=16, specific_neuron=None):
    activation_after_tanh = axis_shift(epi_activation)
    neurons_fft = []
    for layer in range(len(activation_after_tanh)):
        layer_fft = []
        for neuron in range(len(activation_after_tanh[layer])):
            if specific_neuron is None:
                fft_ret, phase = draw_origin_and_fft(
                    activation_after_tanh[layer][neuron],
                    label="neuron_{}_{}".format(layer, neuron),
                    save_figure=save_figure,
                    for_neuron=True,
                    n_fft=n_fft
                )
                layer_fft.append({
                    "fft_amplitude": fft_ret,
                    "fft_phase": phase,
                })
            elif specific_neuron is not None:
                assert isinstance(specific_neuron, list), "Use list [(layer, neuron index), (),...]"
                if (layer, neuron) in specific_neuron:
                    fft_ret, phase = draw_origin_and_fft(
                        activation_after_tanh[layer][neuron],
                        label="neuron_{}_{}".format(layer, neuron),
                        save_figure=save_figure,
                        for_neuron=True,
                        n_fft=n_fft
                    )
                    layer_fft.append({
                        "fft_amplitude": fft_ret,
                        "fft_phase": phase,
                    })
                else:
                    layer_fft.append({
                        "fft_amplitude": np.inf,
                        "fft_phase": np.inf,
                    })

        neurons_fft.append(layer_fft)
    return neurons_fft, activation_after_tanh


def analyze_observation(epi_observation, save_figure=False, n_fft=16, specific_obs=None):
    obs_per_step = np.array(epi_observation)
    per_obs_dim = np.moveaxis(obs_per_step, 0, -1)
    obs_fft = []
    for dim in range(len(per_obs_dim)):
        if specific_obs is None:
            fft_ret, phase = draw_origin_and_fft(
                per_obs_dim[dim],
                label="obs_dim_{}".format(dim),
                save_figure=save_figure,
                for_neuron=False,
                n_fft=n_fft
            )
            obs_fft.append({"fft_amplitude": fft_ret, "fft_phase": phase})
        elif specific_obs is not None:
            if dim in specific_obs:
                fft_ret, phase = draw_origin_and_fft(
                    per_obs_dim[dim],
                    label="obs_dim_{}".format(dim),
                    save_figure=save_figure,
                    for_neuron=False,
                    n_fft=n_fft
                )
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
        fft_ret, phase = draw_origin_and_fft(
            per_action_dim[dim],
            label="action_dim_{}".format(dim),
            save_figure=save_figure,
            for_neuron=False,
            n_fft=n_fft
        )
        action_fft.append({"fft_amplitude": fft_ret, "fft_phase": phase})
    return action_fft, per_action_dim


def do_policy_dissection(collect_episodes, specific_neuron=None, specific_obs=None):
    n_fft = 32
    # assert not os.path.exists("dissection"), "please save previous result"
    # os.makedirs("dissection")
    ckpt_ret = {}
    for k, epi_data in enumerate(collect_episodes):
        print("===== Dissect episode {} =====".format(k))
        epi_activation = epi_data["neuron_activation"]
        observations = epi_data["observations"]

        neurons_fft, origin_neuron = analyze_neuron(epi_activation, n_fft=n_fft, specific_neuron=specific_neuron)

        obs_fft, origin_obs = analyze_observation(observations, n_fft=n_fft, specific_obs=specific_obs)

        ret_obs = get_most_relevant_neuron(
            target_dim_name="obs", neurons_activation_fft=neurons_fft, epi_target_dims_fft=obs_fft
        )["obs_analysis"]
        this_epi_frequency_error = ret_obs
        this_epi_frequency_error["seed"] = k
        ckpt_ret[k] = this_epi_frequency_error
    return ckpt_ret