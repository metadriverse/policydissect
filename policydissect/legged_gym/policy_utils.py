import numpy as np


elu = lambda Z: np.where(Z>0, Z, np.exp(Z)-1)


def control_neuron_activation(layer_out_put, layer, conditional_control_map, command, legged=False):
    x = layer_out_put
    if command in conditional_control_map.keys():
        if layer in conditional_control_map[command]:
            for neuron, activation_score in conditional_control_map[command][layer]:
                if legged:
                    x[neuron] = activation_score
                else:
                    x[0][neuron] = activation_score


def ppo_inference_tf(weights,
                     obs,
                     hidden_layer_num,
                     conditional_control_map,
                     command,
                     deterministic=False,
                     activation="tanh"):
    step_activation_value = []
    obs = obs.reshape(1, -1)
    x = obs
    activate_func = np.tanh if activation == "tanh" else relu
    for layer in range(1, hidden_layer_num + 1):
        x = np.matmul(x, weights["default_policy/fc_{}/kernel".format(layer)]) + weights[
            "default_policy/fc_{}/bias".format(layer)]
        before_tanh = x
        x = activate_func(x)
        control_neuron_activation(x, layer - 1, conditional_control_map, command)
        step_activation_value.append({"after_tanh": x, "before_tanh": before_tanh})

    x = np.matmul(x, weights["default_policy/fc_out/kernel"]) + weights["default_policy/fc_out/bias"]
    x = x.reshape(-1)
    mean, log_std = np.split(x, 2)
    if deterministic:
        return mean
    std = np.exp(log_std)
    return np.random.normal(mean, std)


def ppo_inference_torch(weights, obs,
                        conditional_control_map,
                        command,
                        deterministic=False,
                        activation="elu",
                        tanh_action=False,
                        print_value=False):
    if print_value:
        print("===== numpy =====")
    step_activation_value = []
    activate_func = elu if activation == "elu" else np.tanh
    obs = obs.reshape(1, -1)
    x = obs[0]
    layers = ["actor.0", "actor.2", "actor.4", "actor.6"]
    for layer_index, layer in enumerate(layers):
        x = np.matmul(weights["{}.weight".format(layer)], x) + weights[
            "{}.bias".format(layer)]
        if layer_index < len(layers) - 1:
            x = activate_func(x)
            control_neuron_activation(x, layer_index, conditional_control_map, command, legged=True)
            step_activation_value.append({"after_tanh": [x], "before_tanh": [x]})
        if print_value:
            print(layer_index, ":", x)

    x = x.reshape(-1)
    mean, std = np.tanh(x) if tanh_action else x, weights["std"]
    if deterministic:
        return mean, step_activation_value
    action = np.random.normal(mean, std)
    ret = action
    return ret, step_activation_value


def _normalize_obs(obs_mid, obs_scale, obs):
    ret = (obs - obs_mid) / obs_scale
    return ret
