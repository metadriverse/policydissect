import gym
import numpy as np
from metadrive.examples.ppo_expert.numpy_expert import ckpt_path

from policydissect.metadrive.metadrive_env import MetaDriveEnv
from policydissect.utils.policy import ppo_inference_tf


class HRLSafeMetaDriveEnv(MetaDriveEnv):
    PPO_EXPERT_CONDITIONAL_CONTROL_MAP = {"Left Lane Change": {0: [(123, 8.5)]},
                                          "Right Lane Change": {0: [(123, -8)]},
                                          "Brake": {0: [(249, -20)]}}

    def __init__(self, config=None):
        config = config or {}
        if "action_repeat" in config:
            action_repeat = config["action_repeat"]
            config.pop("action_repeat")
        else:
            action_repeat = 10
        if "use_step_reward" in config:
            self.use_step_reward = config["use_step_reward"]
            config.pop("use_step_reward")
        else:
            self.use_step_reward = False

        if "crash_done" in config:
            self.crash_done = config["crash_done"]
            config.pop("crash_done")
        else:
            self.crash_done = False

        default_config = dict(
            use_render=False,
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
    def observation_space(self):
        return gym.spaces.Box(-1, 1, (super(HRLSafeMetaDriveEnv, self).observation_space.shape[0] + 4,), dtype=np.float64)

    @property
    def action_space(self) -> gym.Space:
        return gym.spaces.Discrete(4)

    def reset(self, *args, **kwargs):
        ret = super(MetaDriveEnv, self).reset(*args, **kwargs)
        self.last_o = ret
        self.command = "Forward"
        index = self.actions.index(self.command)
        return np.concatenate([ret, np.eye(4)[index]], axis=-1)

    def step(self, action):
        ori_action = action
        self.command = self.actions[action]
        self.step_info = []
        # print(command)
        for i in range(self.action_repeat):
            action = ppo_inference_tf(self.policy, self.last_o, 2, self.PPO_EXPERT_CONDITIONAL_CONTROL_MAP,
                                      self.command)
            o, r, d, i = super(HRLSafeMetaDriveEnv, self).step(action)
            self.step_info.append(i)
            self.last_o = o
            if self.config["use_render"]:
                self.render(text={"command": self.command, "step:": self.engine.episode_step})
        infos = self._merge_info()
        done = infos["arrive_dest"] or infos["out_of_road"]
        if self.crash_done:
            done |= infos["crash_vehicle"] or infos["crash_object"] or infos["crash"]
        reward = infos["step_reward"] if self.use_step_reward else 1
        if infos["out_of_road"] or infos["crash_vehicle"] or infos["crash_object"]:
            reward -= 1 * self.action_repeat if self.use_step_reward else 1
        if self.command=="Brake":
            reward /= 2
        return np.concatenate([o, np.eye(4)[int(ori_action)]], axis=-1), reward, done, infos

    def _merge_info(self):
        ret = {}

        ret["step_reward"] = self._sum_info("step_reward")
        ret["cost"] = self._sum_info("cost")
        ret["step_energy"] = self._sum_info("step_energy")

        ret["velocity"] = self._mean_info("velocity")
        ret["steering"] = self._mean_info("steering")
        ret["acceleration"] = self._mean_info("acceleration")

        ret["episode_energy"] = self._max_info("episode_energy")
        ret["total_cost"] = self._max_info("total_cost")
        ret["episode_reward"] = self._max_info("episode_reward")
        ret["episode_length"] = self._max_info("episode_length")
        ret["overtake_vehicle_num"] = self._max_info("overtake_vehicle_num")

        ret["crash_vehicle"] = self._or_info("crash_vehicle")
        ret["crash_object"] = self._or_info("crash_object")
        ret["crash_building"] = self._or_info("crash_building")
        ret["out_of_road"] = self._or_info("out_of_road")
        ret["arrive_dest"] = self._or_info("arrive_dest")
        ret["max_step"] = self._or_info("max_step")
        ret["crash"] = self._or_info("crash")

        return ret

    def _sum_info(self, key):
        ret = 0
        for i in self.step_info:
            ret += i[key]
        return ret

    def _or_info(self, key):
        ret = False
        for i in self.step_info:
            ret |= i[key]
        return ret

    def _mean_info(self, key):
        ret = []
        for i in self.step_info:
            ret.append(i[key])
        return np.mean(ret)

    def _max_info(self, key):
        ret = []
        for i in self.step_info:
            ret.append(i[key])
        return np.max(ret)


if __name__ == "__main__":
    env = HRLSafeMetaDriveEnv(
        config={"use_render": False, "manual_control": True, "use_step_reward": True, "crash_done": True})
    env.reset()
    print(env.observation_space)
    actions = [1, 2, 0, 3]
    while True:
        for action in actions:
            o, r, d, i = env.step(action)
            assert env.observation_space.contains(o)
            print(o[-4:], r, d)
            if d:
                env.reset()
                break
