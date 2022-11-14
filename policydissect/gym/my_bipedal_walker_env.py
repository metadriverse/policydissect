from gym.envs.box2d.bipedal_walker import BipedalWalkerHardcore
import numpy as np
import math

FPS = 50
SCALE = 30.0  # affects how fast-paced the game is, forces should be adjusted as well

MOTORS_TORQUE = 80
MOTORS_TORQUE_BOOST = 5000

SPEED_HIP = 4
SPEED_KNEE = 6
LIDAR_RANGE = 160 / SCALE

INITIAL_RANDOM = 5

HULL_POLY = [(-30, +9), (+6, +9), (+34, +1), (+34, -8), (-30, -8)]
LEG_DOWN = -8 / SCALE
LEG_W, LEG_H = 8 / SCALE, 34 / SCALE

VIEWPORT_W = 600
VIEWPORT_H = 400

TERRAIN_STEP = 14 / SCALE
TERRAIN_LENGTH = 200  # in steps
TERRAIN_HEIGHT = VIEWPORT_H / SCALE / 4
TERRAIN_GRASS = 10  # low long are grass spots, in steps
TERRAIN_STARTPAD = 20  # in steps
FRICTION = 2.5

import time

def wrap_to_pi(angles):
    angles %= 2 * np.pi
    angles -= 2 * np.pi * (angles > np.pi)
    return angles

class MyBipedalWalker(BipedalWalkerHardcore):
    hardcore = False
    count = 0

    def _generate_terrain(self, hardcore):
        GRASS, STUMP, STAIRS, PIT, _STATES_ = range(5)
        state = GRASS
        velocity = 0.0
        y = TERRAIN_HEIGHT
        counter = TERRAIN_STARTPAD
        oneshot = False
        self.terrain = []
        self.terrain_x = []
        self.terrain_y = []
        for i in range(TERRAIN_LENGTH):
            x = i * TERRAIN_STEP
            self.terrain_x.append(x)

            if state == GRASS and not oneshot:
                velocity = 0.8 * velocity + 0.01 * np.sign(TERRAIN_HEIGHT - y)
                # if i > TERRAIN_STARTPAD:
                #     velocity += self.np_random.uniform(-1, 1) / SCALE  # 1
                y += velocity

            oneshot = False
            self.terrain_y.append(y)
            counter -= 1
            if counter == 0:
                counter = self.np_random.randint(TERRAIN_GRASS / 2, TERRAIN_GRASS)
                if state == GRASS and hardcore:
                    state = self.np_random.randint(1, _STATES_)
                    oneshot = True
                else:
                    state = GRASS
                    oneshot = True

        self.terrain_poly = []
        for i in range(TERRAIN_LENGTH - 1):
            poly = [
                (self.terrain_x[i], self.terrain_y[i]),
                (self.terrain_x[i + 1], self.terrain_y[i + 1]),
            ]
            self.fd_edge.shape.vertices = poly
            t = self.world.CreateStaticBody(fixtures=self.fd_edge)
            color = (0.3, 1.0 if i % 2 == 0 else 0.8, 0.3)
            t.color1 = color
            t.color2 = color
            self.terrain.append(t)
            color = (0.4, 0.6, 0.3)
            poly += [(poly[1][0], 0), (poly[0][0], 0)]
            self.terrain_poly.append((poly, color))
        self.terrain.reverse()

    @property
    def command(self):
        return self.viewer.window.command if self.viewer is not None else "right"

    def render(self, mode="human"):
        from policydissect.gym.pyglet_rendering import MyViewer
        if self.viewer is None:
            self.viewer = MyViewer(1200, 800)
        super(MyBipedalWalker, self).render(mode)
        if self.viewer.window.need_reset:
            self.reset()
            self.viewer.window.command = "right"
            self.viewer.window.need_reset = False

    def step(self, action):
        # time.sleep(0.01)
        # self.hull.ApplyForceToCenter((0, 20), True) -- Uncomment this to receive a bit of stability help
        control_speed = False  # Should be easier as well
        TORQUE = MOTORS_TORQUE if self.command != "left" else MOTORS_TORQUE_BOOST
        if self.command == "down" or self.command == "left" or self.command=="up":
            self.count += 1

        if self.count > 55 and self.command != "right":
            self.viewer.window.command = "right"
            self.count = 0

        if control_speed:
            self.joints[0].motorSpeed = float(SPEED_HIP * np.clip(action[0], -2000, 2000))
            self.joints[1].motorSpeed = float(SPEED_KNEE * np.clip(action[1], -2000, 2000))
            self.joints[2].motorSpeed = float(SPEED_HIP * np.clip(action[2], -2000, 2000))
            self.joints[3].motorSpeed = float(SPEED_KNEE * np.clip(action[3], -2000, 2000))
        else:
            self.joints[0].motorSpeed = float(SPEED_HIP * np.sign(action[0]))
            self.joints[0].maxMotorTorque = float(TORQUE * np.clip(np.abs(action[0]), 0, 2000))
            self.joints[1].motorSpeed = float(SPEED_KNEE * np.sign(action[1]))
            self.joints[1].maxMotorTorque = float(TORQUE * np.clip(np.abs(action[1]), 0, 2000))
            self.joints[2].motorSpeed = float(SPEED_HIP * np.sign(action[2]))
            self.joints[2].maxMotorTorque = float(TORQUE * np.clip(np.abs(action[2]), 0, 2000))
            self.joints[3].motorSpeed = float(SPEED_KNEE * np.sign(action[3]))
            self.joints[3].maxMotorTorque = float(TORQUE * np.clip(np.abs(action[3]), 0, 2000))

        self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)

        pos = self.hull.position
        vel = self.hull.linearVelocity

        for i in range(10):
            self.lidar[i].fraction = 1.0
            self.lidar[i].p1 = pos
            self.lidar[i].p2 = (
                pos[0] + math.sin(1.5 * i / 10.0) * LIDAR_RANGE,
                pos[1] - math.cos(1.5 * i / 10.0) * LIDAR_RANGE,
            )
            self.world.RayCast(self.lidar[i], self.lidar[i].p1, self.lidar[i].p2)

        state = [
            wrap_to_pi(self.hull.angle),  # Normal angles up to 0.5 here, but sure more is possible.
            2.0 * self.hull.angularVelocity / FPS,
            0.3 * vel.x * (VIEWPORT_W / SCALE) / FPS,  # Normalized to get -1..1 range
            0.3 * vel.y * (VIEWPORT_H / SCALE) / FPS,
            self.joints[0].angle,
            # This will give 1.1 on high up, but it's still OK (and there should be spikes on hiting the ground, that's normal too)
            self.joints[0].speed / SPEED_HIP,
            self.joints[1].angle + 1.0,
            self.joints[1].speed / SPEED_KNEE,
            1.0 if self.legs[1].ground_contact else 0.0,
            self.joints[2].angle,
            self.joints[2].speed / SPEED_HIP,
            self.joints[3].angle + 1.0,
            self.joints[3].speed / SPEED_KNEE,
            1.0 if self.legs[3].ground_contact else 0.0,
        ]
        state += [l.fraction for l in self.lidar]
        assert len(state) == 24

        self.scroll = pos.x - VIEWPORT_W / SCALE / 5

        shaping = (
            130 * pos[0] / SCALE
        )  # moving forward is a way to receive reward (normalized to get 300 on completion)
        shaping -= 5.0 * abs(state[0])  # keep head straight, other than that and falling, any behavior is unpunished

        reward = 0
        if self.prev_shaping is not None:
            reward = shaping - self.prev_shaping
        self.prev_shaping = shaping

        for a in action:
            reward -= 0.00035 * MOTORS_TORQUE * np.clip(np.abs(a), 0, 1)
            # normalized to about -50.0 using heuristic, more optimal agent should spend less

        done = False
        if self.game_over or pos[0] < 0:
            reward = -100
            done = True
        if pos[0] > (TERRAIN_LENGTH - TERRAIN_GRASS) * TERRAIN_STEP:
            done = True
        return np.array(state, dtype=np.float32), reward, done, {}


if __name__ == "__main__":
    env = MyBipedalWalker()
    env.reset()
    while True:
        env.step(env.action_space.sample())
        env.render()
