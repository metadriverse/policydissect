from gym import error
from policydissect.gym.my_walker_env import MyWalker

stop = {
    "left": {
        # 5: [(219, -10)],
        # 0: [(182, -6)]  # for up/down
        0: [(117, -5), (182, -4)]
    },
    "right": {
        # 5: [(219, 12)],
        # 1: [(163, 1)],
        # 0: [(188, -8)] # 129 10 red foot
        # 1: [(224, -2)],
        # 0: [(117, 6)]
    },
    "brake": {2: [(198, 14)]},
    # "straight": {0: [(127, -2.5)]}
}

stiff = {"brake": {0 :[(191, -6), (172, -8)]}}

if __name__ == "__main__":
    env = MyWalker()
    # 11/14 two tigh
    policy = PolicyNeuronActivationVisualization(env,
                                                 custom_ckpt_path="walker.npz",
                                                 save_activation=False,
                                                 deterministic_policy=False,
                                                 metadrive_ppo_expert_obs=False,
                                                 ppo_agent=False,
                                                 policy_activation="tanh",
                                                 hidden_layer_num=2,
                                                 max_step=10000,
                                                 # right knee/tigh left knee tigh (3, -8), (193, -5), (222, 6), (174,6)
                                                 need_update_analysis=False,
                                                 conditional_control_map=stop
                                                 )
    policy.run()
