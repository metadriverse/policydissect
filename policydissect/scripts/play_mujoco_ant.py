from gym import error
from policydissect.gym.my_ant_env import MyAntEnv

ANT_2_layer_15_ckpt_930 = {
                              "left": {
                                  # 5: [(219, -10)],
                                  # 1: [(163, -2)],
                                  0: [(196, -2), (24, -2), (197, 8)],
                              },
                              "right": {
                                  # 5: [(219, 12)],
                                  # 1: [(163, 1)],
                                  0: [(196, 3), (24, 3), (197, -1)]
                              },
                              "brake": {0: [(156, -20)]},
                              # "straight": {0: [(127, -2.5)]}
                          },

if __name__ == "__main__":
    env = MyAntEnv(random_reset_position=True)
    # 13/14 x/y velocity
    # 16/17/18 x/y/z angular velocity
    # 1/2/3/4 x/y/z/w orientation quarian
    # 14 we only control the orientaiton and y velocity, which are the most useful goal for y drection control
    policy = PolicyNeuronActivationVisualization(env,
                                                 custom_ckpt_path="ant_sac_2_layer.npz",
                                                 save_activation=False,
                                                 deterministic_policy=False,
                                                 metadrive_ppo_expert_obs=False,
                                                 ppo_agent=False,
                                                 policy_activation="tanh",
                                                 hidden_layer_num=2,
                                                 max_step=100000,
                                                 # right knee/tigh left knee tigh (3, -8), (193, -5), (222, 6), (174,6)
                                                 conditional_control_map={
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
                                                     "brake": {0: [(156, -20)]},
                                                     # "straight": {0: [(127, -2.5)]}
                                                 },
                                                 need_update_analysis=False
                                                 # conditional_control_map=None
                                                 )
    policy.run()
