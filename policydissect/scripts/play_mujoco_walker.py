import numpy as np
import os

from policydissect import PACKAGE_DIR
from policydissect.gym.my_walker_env import MyWalker
from policydissect.utils.policy import sac_inference_tf

WALKER_MAP = {
    "up": {
        0: [(191, -6), (172, -8)]
    },  # stiff
    "brake": {
        0: [(117, -2.8), (182, -7.5)]
    },
}

if __name__ == "__main__":
    """
    KEY_R: reset
    KEY_A: stop
    KEY_W: freeze red knee
    KEY_D: restore running
    """

    env = MyWalker()
    # 11/14 two tigh
    policy_weights = np.load(os.path.join(PACKAGE_DIR, "weights", "walker.npz"))
    # 0-21 position
    # x/y 22, 23
    # 25/26/27 x/y/z angular
    # 1/2/3/4 x/y/z/w orientation quarian
    o = env.reset()
    while True:
        o, r, d, i = env.step(sac_inference_tf(policy_weights, o, 2, WALKER_MAP, env.command))
        env.render()
        if d:
            o = env.reset()
            env.viewer.keyboard_action = "restore"
