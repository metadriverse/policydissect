from policydissect.gym.my_ant_env import MyAntEnv
import os
from policydissect import PACKAGE_DIR
import numpy as np
from policydissect.utils.policy import sac_inference_tf

ANT_MAP = {
              "left": {
                  # 5: [(219, -10)],
                  # 1: [(163, -2)],
                  0: [(196, -2), (24, -2), (197, 8)]  # for up/down
              },
              "right": {
                  # 5: [(219, 12)],
                  # 1: [(163, 1)],
                  0: [(196, 3), (24, 3), (197, -1)]
                  # 0: [(24, -43)] # for rotation
              },
              "brake": {
                  0: [(156, -20)]
              },
              # "straight": {0: [(127, -2.5)]}
          },

if __name__ == "__main__":
    env = MyAntEnv(random_reset_position=True)
    # 13/14 x/y velocity
    # 16/17/18 x/y/z angular velocity
    # 1/2/3/4 x/y/z/w orientation quarian
    # 14 we only control the orientaiton and y velocity, which are the most useful goal for y drection control
    policy_weights = np.load(os.path.join(PACKAGE_DIR, "weights", "ant.npz"))
    # 0-21 position
    # x/y 22, 23
    # 25/26/27 x/y/z angular
    # 1/2/3/4 x/y/z/w orientation quarian
    o = env.reset()
    while True:
        o, r, d, i = env.step(sac_inference_tf(policy_weights, o, 2, ANT_MAP, env.command))
        env.render()
        if d:
            o = env.reset()
