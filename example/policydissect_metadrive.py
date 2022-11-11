import numpy as np
import time
from policydissect.policydissect import do_policy_dissection
from policydissect.metadrive.metadrive_env import SafeMetaDriveEnv
from policydissect.utils.policy import ppo_inference_tf
import os
import pickle

if __name__ == "__main__":
    from metadrive.examples.ppo_expert.numpy_expert import ckpt_path
    import random

    seed = 0
    random.seed(seed)
    np.random.seed(seed)

    policy_func = ppo_inference_tf
    seed_num = 1
    max_step = 1500

    env = SafeMetaDriveEnv(
        dict(
            use_render=False,
            accident_prob=0.,
            traffic_density=0.,
            random_spawn_lane_index=False,
            map="COCX",
            environment_num=1,
            start_seed=30,
            vehicle_config=dict(
                lidar=dict(num_lasers=240, distance=50, num_others=4, gaussian_noise=0.0, dropout_prob=0.0)
            ),
        )
    )

    print("===== Do Policy Dissection for on ckpt =====")
    path = ckpt_path
    weights = np.load(ckpt_path)
    collected_episodes = []
    for seed in range(seed_num):
        o = env.reset()
        episode_activation_values = []
        episode_observations = [o]
        current_step = 0
        total_r = 0

        while True:
            action, activation = policy_func(weights, o, hidden_layer_num=2, conditional_control_map={}, command="")
            o, r, d, i = env.step(action)
            episode_activation_values.append(activation)
            current_step += 1
            total_r += r
            if d or current_step > max_step:
                collected_episodes.append(
                    dict(neuron_activation=episode_activation_values, observations=episode_observations)
                )
                print("Finish seed: {}, reward: {}, Success: {}".format(seed, total_r, i["arrive_dest"]))
                break
            episode_observations.append(o)
    env.close()
    pd_ret = do_policy_dissection(collected_episodes)
    with open("{}.pkl".format("metadrive_ret"), "wb+") as file:
        pickle.dump(pd_ret, file)
