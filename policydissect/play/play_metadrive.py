import numpy as np
from metadrive.examples.ppo_expert.numpy_expert import ckpt_path

from policydissect.utils.metadrive_env import MetaDriveEnv
from policydissect.utils.policy import ppo_inference_tf

# neuron at layer 0, index 123 is for lateral control
# neuron at layer 0, index 249 is for speed control
PPO_EXPERT_CONDITIONAL_CONTROL_MAP = {
    "Left Lane Change": {
        0: [(123, 8.5)]
    },
    "Right Lane Change": {
        0: [(123, -8)]
    },
    "Brake": {
        0: [(249, -20)]
    }
}

if __name__ == "__main__":
    builtin_ppo = np.load(ckpt_path)
    env = MetaDriveEnv(
        dict(
            use_render=True,
            start_seed=500,
            accident_prob=0.8,
            traffic_density=0.1,
            environment_num=20,
            vehicle_config=dict(
                lidar=dict(num_lasers=240, distance=50, num_others=4, gaussian_noise=0.0, dropout_prob=0.0)
            ),
        )
    )
    o = env.reset()
    while True:
        o, r, d, i = env.step(ppo_inference_tf(builtin_ppo, o, 2, PPO_EXPERT_CONDITIONAL_CONTROL_MAP, env.command))
        env.render(
            text={
                "w": "Lane Follow",
                "s": "Brake",
                "a": "Left Lane Change",
                "d": "Right Lane Change",
                "r": "Reset",
                "f": "Unlimited FPS",
                "Current Command": env.command
            }
        )
        if d:
            o = env.reset()
