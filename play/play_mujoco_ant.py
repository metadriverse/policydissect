from policydissect.gym.my_ant_env import MyAntEnv
import os
from policydissect import PACKAGE_DIR
import numpy as np
from policydissect.utils.policy import sac_inference_tf

ANT_MAP = {
    "up": {
        0: [(196, -3), (24, -7), (197, 6)]  # for moving up
    },
    "down": {
        0: [(196, 3), (24, 3), (197, -1)]
    },  # for moving down
    "brake": {
        0: [(156, -20)]
    },  # stop
    "restore": {},  # restore
    "rotation": {
        0: [(24, -43)]
    }  # for rotation
}

if __name__ == "__main__":
    """
    KEY_W: move up
    KEY_A: move left
    KEY_S: move down
    KEY_D: move right
    KEY_Q: rotation
    KEY_R: reset
    """
    env = MyAntEnv(random_reset_position=True)
    policy_weights = np.load(os.path.join(PACKAGE_DIR, "weights", "ant.npz"))
    o = env.reset()
    while True:
        o, r, d, i = env.step(sac_inference_tf(policy_weights, o, 2, ANT_MAP, env.command))
        env.render()
        if d:
            o = env.reset()
            env.viewer.keyboard_action = "restore"
