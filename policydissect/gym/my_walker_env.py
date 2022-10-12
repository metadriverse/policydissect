from policydissect.gym.my_ant_env import CustomViewer
from gym import error
from gym.envs.mujoco.walker2d_v3 import Walker2dEnv

try:
    import mujoco_py
except ImportError as e:
    raise error.DependencyNotInstalled(
        "{}. (HINT: you need to install mujoco_py, and also perform the setup instructions here: https://github.com/openai/mujoco-py/.)"
        .format(e)
    )
import mujoco_py


class MyWalker(Walker2dEnv):
    def __init__(self, place_holder={}, *args, **kwargs):
        super(MyWalker, self).__init__(*args, **kwargs)

    # def reset_model(self):
    #     noise_low = -self._reset_noise_scale
    #     noise_high = self._reset_noise_scale
    #
    #     qpos = self.init_qpos + self.np_random.uniform(
    #         low=noise_low, high=noise_high, size=self.model.nq
    #     )
    #
    #     qpos[6] = self.np_random.uniform(low=-6.28, high=6.28)
    #
    #     qvel = self.init_qvel + self.np_random.uniform(
    #         low=noise_low, high=noise_high, size=self.model.nv
    #     )
    #     self.set_state(qpos, qvel)
    #
    #     observation = self._get_obs()
    #     return observation

    def _get_viewer(self, mode):
        self.viewer = self._viewers.get(mode)
        if self.viewer is None:
            if mode == "human":
                self.viewer = CustomViewer(self.sim)
            elif mode == "rgb_array" or mode == "depth_array":
                self.viewer = mujoco_py.MjRenderContextOffscreen(self.sim, -1)

            self.viewer_setup()
            self._viewers[mode] = self.viewer

            self.viewer.cam.lookat[0] = 10  # x,y,z offset from the object (works if trackbodyid=-1)
            self.viewer.cam.lookat[1] = 0
            self.viewer.cam.lookat[2] = 2
            self.viewer.cam.distance = 15

            self.viewer.cam.elevation = 0  # camera rotation around the axis in the plane going through the frame origin (if 0 you just see a line)
            # self.viewer.cam.azimuth = 0
        return self.viewer

    @property
    def command(self):
        return self.viewer.keyboard_action if self.viewer is not None else "straight"

    def step(self, action):
        if self.viewer is not None and self.viewer.need_reset:
            self.reset()
            self.viewer.need_reset = False
        return super(MyWalker, self).step(action)


if __name__ == "__main__":
    env = MyWalker()
    env.reset()
    for i in range(100000):

        env.step(env.action_space.sample())
        env.render()
        if i % 100 == 0:
            env.reset()
