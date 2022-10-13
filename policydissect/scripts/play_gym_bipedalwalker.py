import numpy as np
import os

from policydissect import PACKAGE_DIR
from policydissect.gym.my_bipedal_walker_env import MyBipedalWalker
from policydissect.utils.policy import sac_inference_tf

jump = {
    "left": {
        0: [(150, 10), (189, 10), (249, 10), (235, 10)]
    },  # key_a: stand up from split
    "down": {
        0: [(202, 40)]
    },  # key_s: restore running after jumping
    "up": {
        0: [(32, 12), (98, -8)]
    }  # key_w: jump
}

if __name__ == "__main__":
    env = MyBipedalWalker()
    policy_weights = np.load(os.path.join(PACKAGE_DIR, "weights", "bipedal_walker.npz"))
    # 0-21 position
    # x/y 22, 23
    # 25/26/27 x/y/z angular
    # 1/2/3/4 x/y/z/w orientation quarian
    o = env.reset()
    while True:
        o, r, d, i = env.step(sac_inference_tf(policy_weights, o, 2, jump, env.command))
        env.render()
        if d:
            o = env.reset()
