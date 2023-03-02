from isaacgym import gymapi  # We have to put this line here!
from policydissect.legged_gym.envs import *  # We have to put this line here!
from policydissect.legged_gym.utils import get_args
from policydissect.utils.isaacgym_utils import play_anymal

if __name__ == '__main__':
    """
    This agent is trained to move forward. After policy dissection, it can change moving direction with human's help.
    This experiment is designed to replicate the results reported in Pybullet-A1 experiment, where the A1 robot can only
    move forward, but can change moving direction by activating specific neurons.
    
        
    Coordinates (Right hand):
                    ^ x
                    |
                    |
                    |
    y <-------------|
    
    """
    """
    Keymap:
    - KEY_W:Forward
    - KEY_A:Left
    - KEY_S:Stop
    - KEY_D:Right
    - KEY_R:Reset
    """
    forward_anymal = {
        "Right": {
            0: [(67, -5)],
        },
        "Left": {
            0: [(67, 3)]
        },
        # 1
        "Forward": {},
        "Stop": {
            1: [(17, -8)],
        }
    }
    args = get_args()
    args.num_envs = 1
    args.task = "anymal_c_flat_forward"
    activation = "tanh"
    play_anymal(args, activation_func=activation, map=forward_anymal, model_name="anymal_forward", parkour=False)
