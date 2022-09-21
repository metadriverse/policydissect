from isaacgym import gymapi
from policydissect.legged_gym.envs import *
from policydissect.legged_gym.utils import get_args
from policydissect.utils.isaacgym_utils import play

if __name__ == '__main__':
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
    args = get_args()
    args.num_envs = 1
    args.task = "cassie"
    activation = "elu"
    play(args, activation_func=activation, map=forward_cassie, model_name="forward_cassie")
