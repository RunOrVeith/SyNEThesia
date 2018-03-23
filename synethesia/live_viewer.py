import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

plt.rcParams['toolbar'] = 'None'


class LiveViewer(object):

    def __init__(self, approx_fps, border_color):
        self.pause_time = 1 / approx_fps
        self.fig = plt.figure()
        plt.tight_layout(pad=0)
        self.fig.patch.set_facecolor(border_color)
        self.fig.canvas.mpl_connect('close_event', self.handle_close)
        self.fig.canvas.mpl_connect('key_press_event', self.toggle_fullscreen)

    def display(self, image_generator):
        im = plt.imshow(np.zeros((1, 1, 3)), animated=True)
        def update(frame, *_):
            im.set_array(frame)
            return im,

        ani = FuncAnimation(self.fig, func=update, frames=image_generator, interval=self.pause_time, blit=True)
        plt.show()

    @staticmethod
    def toggle_fullscreen(event=None):
        if event is None or event.key == "escape":
            mng = plt.get_current_fig_manager()
            mng.full_screen_toggle()

    @staticmethod
    def handle_close(event):
        plt.close("all")
