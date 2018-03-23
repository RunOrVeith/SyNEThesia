import matplotlib.pyplot as plt
import matplotlib.animation as animation
plt.rcParams['toolbar'] = 'None'


class LiveViewer(object):

    def __init__(self, approx_fps, border_color):
        self.pause_time = 1 / approx_fps
        fig, self.img_view = plt.subplots()
        self.img_view.set_frame_on(False)
        self.img_view.get_xaxis().set_visible(False)
        self.img_view.get_yaxis().set_visible(False)
        plt.tight_layout(pad=0)
        fig.patch.set_facecolor(border_color)
        fig.canvas.mpl_connect('close_event', self.handle_close)
        fig.canvas.mpl_connect('key_press_event', self.toggle_fullscreen)

    def __enter__(self):
        plt.ion()
        self.toggle_fullscreen()
        return self

    def __exit__(self, *args, **kwargs):
        plt.ioff()
        plt.close("all")
        # TODO exiting causes a tkinter error during update while it pauses

    def display(self, image):
        self.img_view.imshow(image)
        plt.pause(self.pause_time)

    @staticmethod
    def toggle_fullscreen(event=None):
        if event is None or event.key == "escape":
            mng = plt.get_current_fig_manager()
            mng.full_screen_toggle()

    @staticmethod
    def handle_close(event):
        plt.close("all")
