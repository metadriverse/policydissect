from isaacgym import gymapi
from policydissect.legged_gym.envs import *
from policydissect.legged_gym.utils import get_args
from policydissect.utils.isaacgym_utils import follow_command

if __name__ == '__main__':
    args = get_args(
        [
            {
                "name": "--primitive_activation",
                "action": "store_true",
                "default": False,
                "help": "The way to track the command, default: explicit goal-conditioned control"
            }
        ]
    )
    args.num_envs = 1
    args.task = "anymal_c_flat"
    activation = "tanh"
    follow_command(
        args,
        activation_func=activation,
        layer=0,
        index=121,
        model_name="anymal",
        target_heading_list=[0, 0.4, 0.9, 1.2, 0.9, 0.4, 0., -0.4, -0.9, -1.2, -0.9, -0.4, 0.],
        command_last_time=100,
        trigger_neuron=False
    )
