from metadrive.envs.safe_metadrive_env import SafeMetaDriveEnv


class MetaDriveEnv(SafeMetaDriveEnv):
    """
    Original Out_of_road condition is too strict for policy learning
    """
    command = "Lane Follow"

    def _is_out_of_road(self, vehicle):
        return vehicle.out_of_route

    def setup_engine(self):
        super(MetaDriveEnv, self).setup_engine()
        self.engine.accept("w", self.forward)
        self.engine.accept("s", self.brake)
        self.engine.accept("a", self.left)
        self.engine.accept("d", self.right)

    def forward(self):
        self.command = "Lane Follow"

    def brake(self):
        self.command = "Brake"

    def left(self):
        self.command = "Left Lane Change"

    def right(self):
        self.command = "Right Lane Change"


if __name__ == "__main__":
    env = MetaDriveEnv({"manual_control": True, "use_render": True})
    env.reset()
    while True:
        o, r, d, i = env.step([0, 0])
        if d:
            env.reset()
