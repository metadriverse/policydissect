import gym
import numpy as np
from metadrive.examples.ppo_expert.numpy_expert import ckpt_path

from policydissect.metadrive.metadrive_env import MetaDriveEnv
from policydissect.utils.policy import ppo_inference_tf


class HRLSafeMetaDriveEnv(MetaDriveEnv):
    PPO_EXPERT_CONDITIONAL_CONTROL_MAP = {"Left Lane Change": {0: [(123, 8.5)]},
                                          "Right Lane Change": {0: [(123, -8)]},
                                          "Brake": {0: [(249, -20)]}}

    def __init__(self, config, action_repeat):
        default_config = dict(
            use_render=True,
            start_seed=500,
            accident_prob=0.8,
            traffic_density=0.1,
            environment_num=20,
            vehicle_config=dict(
                lidar=dict(num_lasers=240, distance=50, num_others=4, gaussian_noise=0.0, dropout_prob=0.0)))
        default_config.update(config)
        super(MetaDriveEnv, self).__init__(config=default_config)
        self.action_repeat = action_repeat
        self.last_o = None
        self.actions = list(self.PPO_EXPERT_CONDITIONAL_CONTROL_MAP.keys()) + ["Forward"]
        self.policy = np.load(ckpt_path)
        self.command = "Forward"

    @property
    def action_space(self) -> gym.Space:
        return gym.spaces.Discrete(4)

    def reset(self, *args, **kwargs):
        ret = super(MetaDriveEnv, self).reset(*args, **kwargs)
        self.last_o = ret
        self.command = "Forward"
        return ret

    def step(self, action):
        self.command = self.actions[action]
        # print(command)
        total_r = 0
        for i in range(self.action_repeat):
            action = ppo_inference_tf(self.policy, self.last_o, 2, self.PPO_EXPERT_CONDITIONAL_CONTROL_MAP,
                                      self.command)
            o, r, d, i = super(HRLSafeMetaDriveEnv, self).step(action)
            total_r += r
            self.last_o = o
            if self.config["use_render"]:
                self.render(text={"command": env.command, "step:": env.engine.episode_step})
        return o, total_r, d, i


if __name__ == "__main__":
    env = HRLSafeMetaDriveEnv(config={"use_render": True}, action_repeat=50)
    env.reset()
    actions = [1, 2, 0, 3]
    while True:
        for action in actions:
            o, r, d, i = env.step(action)
            if d:
                env.reset()
                break
