import os.path

import glfw
import imageio
import numpy as np
from gym import error
from gym import utils

from mujoco_py.generated import const

try:
    import mujoco_py
except ImportError as e:
    raise error.DependencyNotInstalled(
        "{}. (HINT: you need to install mujoco_py, and also perform the setup instructions here: https://github.com/openai/mujoco-py/.)".format(
            e
        )
    )
import mujoco_py
from gym.envs.mujoco.ant_v3 import AntEnv
from mujoco_py.mjviewer import MjViewer, MjViewerBasic


class CustomViewer(MjViewer):
    need_reset = False
    keyboard_action = "straight"
    def __init__(self, sim):
        super(CustomViewer, self).__init__(sim)
        # video record is useless
        self._video_path = "F:\\hk\\drivingforce\\drivingforce\\policy_dissection\\video_%07d.mp4"
        self._image_path = "F:\\hk\\drivingforce\\drivingforce\\policy_dissection\\frame_%07d.png"

    def key_callback(self, window, key, scancode, action, mods):
        # always set to fallse

        if key == glfw.KEY_KP_5:
            self.keyboard_action = "brake"
        if key == glfw.KEY_KP_8:
            self.keyboard_action = "straight"
        if key == glfw.KEY_KP_4:
            self.keyboard_action = "left"
        if key == glfw.KEY_KP_6:
            self.keyboard_action = "right"
        if key == glfw.KEY_KP_7:
            self.need_reset = True
        if action != glfw.RELEASE:
            return
        elif key == glfw.KEY_TAB:  # Switches cameras.
            self.cam.fixedcamid += 1
            self.cam.type = const.CAMERA_FIXED
            if self.cam.fixedcamid >= self._ncam:
                self.cam.fixedcamid = -1
                self.cam.type = const.CAMERA_FREE
        elif key == glfw.KEY_H:  # hides all overlay.
            self._hide_overlay = not self._hide_overlay
        elif key == glfw.KEY_SPACE and self._paused is not None:  # stops simulation.
            self._paused = not self._paused
            # Advances simulation by one step.
        elif key == glfw.KEY_RIGHT and self._paused is not None:
            self._advance_by_one_step = True
            self._paused = True
        # elif key == glfw.KEY_V or \
        #         (
        #                 key == glfw.KEY_ESCAPE and self._record_video):  # Records video. Trigers with V or if in progress by ESC.
        #     self._record_video = not self._record_video
        #     if self._record_video:
        #         fps = (1 / self._time_per_render)
        #         self._video_process = Process(target=save_video,
        #                                       args=(self._video_queue, self._video_path % self._video_idx, fps))
        #         self._video_process.start()
        #     if not self._record_video:
        #         self._video_queue.put(None)
        #         self._video_process.join()
        #         self._video_idx += 1
        elif key == glfw.KEY_T:  # capture screenshot
            img = self._read_pixels_as_in_window()
            imageio.imwrite(self._image_path % self._image_idx, img)
            self._image_idx += 1
        elif key == glfw.KEY_I:  # drops in debugger.
            print('You can access the simulator by self.sim')
            import ipdb
            ipdb.set_trace()
        elif key == glfw.KEY_S:  # Slows down simulation.
            self._run_speed /= 2.0
        elif key == glfw.KEY_F:  # Speeds up simulation.
            self._run_speed *= 2.0
        elif key == glfw.KEY_C:  # Displays contact forces.
            vopt = self.vopt
            vopt.flags[10] = vopt.flags[11] = not vopt.flags[10]
        elif key == glfw.KEY_D:  # turn off / turn on rendering every frame.
            self._render_every_frame = not self._render_every_frame
        elif key == glfw.KEY_E:
            vopt = self.vopt
            vopt.frame = 1 - vopt.frame
        elif key == glfw.KEY_R:  # makes everything little bit transparent.
            self._transparent = not self._transparent
            if self._transparent:
                self.sim.model.geom_rgba[:, 3] /= 5.0
            else:
                self.sim.model.geom_rgba[:, 3] *= 5.0
        elif key == glfw.KEY_M:  # Shows / hides mocap bodies
            self._show_mocap = not self._show_mocap
            for body_idx1, val in enumerate(self.sim.model.body_mocapid):
                if val != -1:
                    for geom_idx, body_idx2 in enumerate(self.sim.model.geom_bodyid):
                        if body_idx1 == body_idx2:
                            if not self._show_mocap:
                                # Store transparency for later to show it.
                                self.sim.extras[
                                    geom_idx] = self.sim.model.geom_rgba[geom_idx, 3]
                                self.sim.model.geom_rgba[geom_idx, 3] = 0
                            else:
                                self.sim.model.geom_rgba[
                                    geom_idx, 3] = self.sim.extras[geom_idx]
        elif key in (glfw.KEY_0, glfw.KEY_1, glfw.KEY_2, glfw.KEY_3, glfw.KEY_4):
            self.vopt.geomgroup[key - glfw.KEY_0] ^= 1
        MjViewerBasic.key_callback(self, window, key, scancode, action, mods)


class MyAntEnv(AntEnv):
    def __init__(
            self,
            place_holder={},
            ctrl_cost_weight=0.5,
            contact_cost_weight=5e-4,
            healthy_reward=1.0,
            terminate_when_unhealthy=True,
            healthy_z_range=(0.2, 1.0),
            contact_force_range=(-1.0, 1.0),
            reset_noise_scale=0.1,
            exclude_current_positions_from_observation=True,
            frame_skip=5,
            random_reset_position=True
    ):
        xml_file = os.path.join(os.path.dirname(__file__), "my_ant.xml")
        utils.EzPickle.__init__(**locals())


        self.random_reset_position=random_reset_position
        self._ctrl_cost_weight = ctrl_cost_weight
        self._contact_cost_weight = contact_cost_weight

        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range

        self._contact_force_range = contact_force_range

        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )

        # if model_path.startswith("/"):
        #     fullpath = model_path
        # else:
        #     fullpath = os.path.join(os.path.dirname(__file__), "assets", model_path)
        # if not path.exists(fullpath):
        #     raise IOError("File %s does not exist" % fullpath)
        self.frame_skip = frame_skip
        self.model = mujoco_py.load_model_from_path(xml_file)
        self.sim = mujoco_py.MjSim(self.model)
        self.data = self.sim.data
        self.viewer = None
        self._viewers = {}

        self.metadata = {
            "render.modes": ["human", "rgb_array", "depth_array"],
            "video.frames_per_second": int(np.round(1.0 / self.dt)),
        }

        self.init_qpos = self.sim.data.qpos.ravel().copy()
        self.init_qvel = self.sim.data.qvel.ravel().copy()

        self._set_action_space()

        action = self.action_space.sample()
        observation, _reward, done, _info = self.step(action)
        assert not done

        self._set_observation_space(observation)

        self.seed()

    def _get_viewer(self, mode):
        self.viewer = self._viewers.get(mode)
        if self.viewer is None:
            if mode == "human":
                self.viewer = CustomViewer(self.sim)
            elif mode == "rgb_array" or mode == "depth_array":
                self.viewer = mujoco_py.MjRenderContextOffscreen(self.sim, -1)

            self.viewer_setup()
            self._viewers[mode] = self.viewer

            # self.viewer.cam.lookat[0] =0  # x,y,z offset from the object (works if trackbodyid=-1)
            # self.viewer.cam.lookat[1] = 0
            # self.viewer.cam.lookat[2] = 0
            # self.viewer.cam.distance = 18

            self.viewer.cam.lookat[0] = 12  # x,y,z offset from the object (works if trackbodyid=-1)
            self.viewer.cam.lookat[1] = 0
            self.viewer.cam.lookat[2] = 0
            self.viewer.cam.distance = 18

            self.viewer.cam.elevation = -90  # camera rotation around the axis in the plane going through the frame origin (if 0 you just see a line)
            # self.viewer.cam.azimuth = 0
        return self.viewer

    @property
    def command(self):
        return self.viewer.keyboard_action if self.viewer is not None else "straight"

    def step(self, action):
        if self.viewer is not None and self.viewer.need_reset:
            self.reset()
            self.viewer.need_reset = False
        xy_position_before = self.get_body_com("torso")[:2].copy()
        self.do_simulation(action, self.frame_skip)
        xy_position_after = self.get_body_com("torso")[:2].copy()

        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        x_velocity, y_velocity = xy_velocity

        ctrl_cost = self.control_cost(action)
        contact_cost = self.contact_cost

        forward_reward = max(abs(x_velocity), abs(y_velocity))  # useless, agents always fall back to single modality
        healthy_reward = self.healthy_reward

        rewards = forward_reward + healthy_reward
        costs = ctrl_cost + contact_cost

        reward = rewards - costs
        done = self.done
        observation = self._get_obs()
        info = {
            "reward_forward": forward_reward,
            "reward_ctrl": -ctrl_cost,
            "reward_contact": -contact_cost,
            "reward_survive": healthy_reward,
            "x_position": xy_position_after[0],
            "y_position": xy_position_after[1],
            "distance_from_origin": np.linalg.norm(xy_position_after, ord=2),
            "x_velocity": x_velocity,
            "y_velocity": y_velocity,
            "forward_reward": forward_reward,
        }

        return observation, reward, done, info

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale


        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )

        # random orientation
        if self.random_reset_position:
            qpos[6] = self.np_random.uniform(low=-6.28, high=6.28)

        qvel = self.init_qvel + self._reset_noise_scale * self.np_random.randn(
            self.model.nv
        )
        self.set_state(qpos, qvel)

        observation = self._get_obs()

        return observation

if __name__=="__main__":
    env=MyAntEnv()
    env.reset()
    while True:
        o,r,d,i = env.step(env.action_space.sample())
        env.render()
        if d:
            env.reset()