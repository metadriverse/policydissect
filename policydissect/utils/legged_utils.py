import copy
import random

import torch
from torch import nn

from policydissect.quadrupedal.vision4leg.get_env import get_single_env

import numpy as np


def seed_env(env, seed):
    env.eval()
    env.seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def get_single_hrl_env(env_id, env_param):
    from policydissect.utils.legged_hrl_env import HRLWrapper
    env = get_single_env(env_id, env_param)
    return HRLWrapper(env, repeat=env_param["env_build"]["action_repeat"])


class ControllableMLP(nn.Module):
    def __init__(
            self,
            weights,
            conditional_control_map,
            activation_func=nn.Tanh):
        super().__init__()
        self.conditional_control_map = conditional_control_map
        self.activation_func = activation_func
        layers = ["base.seq_fcs.0", "base.seq_fcs.2", "seq_append_fcs.0", "seq_append_fcs.2", "seq_append_fcs.4"]
        self.layer_output = []
        self.log_std = torch.exp(torch.from_numpy(weights["logstd"]))
        for layer_index, layer in enumerate(layers):
            fcs = []
            original_w = copy.deepcopy(weights["{}.weight".format(layer)]).astype(np.float64)
            original_b = copy.deepcopy(weights["{}.bias".format(layer)]).astype(np.float64)
            fc = nn.Linear(original_w.shape[0], original_w.shape[1], device="cpu").double()
            with torch.no_grad():
                fc.weight.data = torch.from_numpy(original_w)
                fc.bias.data = torch.from_numpy(original_b)
            # fc.to(dtype=torch.float64)
            fcs.append(fc)
            if layer_index < len(layers) - 1:
                fcs.append(activation_func())
            self.layer_output.append(nn.Sequential(*fcs))

        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x, command, deterministic=True, print_value=False):
        if print_value:
            print("===== torch =====")
        for layer_index, fc in enumerate(self.layer_output):
            x = fc(x)
            if layer_index < len(self.layer_output) - 1 and command in self.conditional_control_map and layer_index in \
                    self.conditional_control_map[command]:
                for neuron, activation_score in self.conditional_control_map[command][layer_index]:
                    x[neuron] = activation_score
            if print_value:
                print(layer_index, ":" , x)
        mean = torch.tanh(x)
        if deterministic:
            return mean
        else:
            return torch.distributions.Normal(mean, self.log_std).sample()
