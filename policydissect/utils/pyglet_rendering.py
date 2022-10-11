from gym.envs.classic_control.rendering import Viewer, get_display, Transform, glEnable, GL_BLEND, glBlendFunc, \
    GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA
import pyglet
key = pyglet.window.key

def get_window(width, height, display, **kwargs):
    """
    Will create a pyglet window from the display specification provided.
    """
    screen = display.get_screens()  # available screens
    config = screen[0].get_best_config()  # selecting the first screen
    context = config.create_context(None)  # create GL context

    return MyWindow(
        width=width,
        height=height,
        display=display,
        config=config,
        context=context,
        **kwargs
    )

class MyWindow(pyglet.window.Window):
    command = "straight"
    need_reset = False

    def on_key_press(self, symbol, modifiers):
        if symbol == key.NUM_4:
            self.command = "left"
        elif symbol == key.NUM_6:
            self.command = "right"
        elif symbol == key.NUM_5:
            self.command = "brake"
        elif symbol == key.NUM_8:
            self.command = "straight"

        if symbol == key.R:
            self.need_reset = True

class MyViewer(Viewer):
    def __init__(self, width, height, display=None):
        display = get_display(display)

        self.width = width
        self.height = height
        self.window = get_window(width=width, height=height, display=display)
        self.window.on_close = self.window_closed_by_user
        self.isopen = True
        self.geoms = []
        self.onetime_geoms = []
        self.transform = Transform()

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)