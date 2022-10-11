from policydissect.gym.my_bipedal_walker_env import MyBipedalWalker, BigForceMyPedalWalker

big_force = {
    "left": {
        # 5: [(219, -10)],
        # 1: [(163, -2)],

        0: [(24, -20)]
        # 0: [(98, 20)] #nice
    },
    "right": {
        # 5: [(219, 12)],
        # 1: [(163, 1)],
        # 0: [(186, 8)],
        0: [(24, 20)]
        # 0: [(24, -45)] # for rotation
    },
    "brake": {0: [(224, -20)]}  # for up/down}
    # "straight": {0: [(127, -2.5)]}
}

torque_force = {
                "left": {0: [(202, -10)]},
                "brake": {0: [(150, 10), (189, 10),(249, 10), (235, 10)]}

                }

jump = {
    "left": {
        # 5: [(219, -10)],
        # 1: [(163, -2)],

        0: [(202, -20)]
        # 0: [(98, 20)] #nice
    },
    "right": {
        # 5: [(219, 12)],
        # 1: [(163, 1)],
        # 0: [(186, 8)],
        0: [(202, 40)]
        # 0: [(24, -45)] # for rotation
    },
    "brake": {0: [(32, 12), (98, -8)]}  # for up/down}
    # "straight": {0: [(127, -2.5)]}
}

if __name__ == "__main__":
    env = MyBipedalWalker()
    # 0-21 position
    # x/y 22, 23
    # 25/26/27 x/y/z angular
    # 1/2/3/4 x/y/z/w orientation quarian
    policy = PolicyNeuronActivationVisualization(env,
                                                 custom_ckpt_path="bipedal_walker.npz",
                                                 save_activation=False,
                                                 deterministic_policy=False,
                                                 metadrive_ppo_expert_obs=False,
                                                 ppo_agent=False,
                                                 policy_activation="tanh",
                                                 hidden_layer_num=2,
                                                 max_step=10000,
                                                 # right knee/tigh left knee tigh (3, -8), (193, -5), (222, 6), (174,6)
                                                 need_update_analysis=True,
                                                 conditional_control_map=torque_force,
                                                 )
    policy.run()
