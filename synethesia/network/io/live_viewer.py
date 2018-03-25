import struct

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import pyaudio

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



class AudioRecorder(object):

    def __init__(self, feature_extractor, frames_per_buffer=2048):
        self.feature_extractor = feature_extractor
        self.device_idx = None
        self.frames_per_buffer = frames_per_buffer
        self.format = pyaudio.paInt16
        self.channels = None
        self.sampling_rate = None
        self.stream = None
        self.feature_dim = 41 # TODO get this properly somehow
        self.audio_controller = pyaudio.PyAudio()
        self.setup_device()

    def setup_device(self):
        global_info = self.audio_controller.get_host_api_info_by_index(0)
        self.device_idx = int(global_info.get("defaultInputDevice"))
        device_info = self.audio_controller.get_device_info_by_host_api_device_index(0, self.device_idx)
        self.sampling_rate = int(device_info.get("defaultSampleRate"))
        self.channels = int(device_info.get("maxInputChannels"))

    def print_input_device_info(self):
        info = self.audio_controller.get_host_api_info_by_index(0)
        numdevices = info.get('deviceCount')
        print("-------Available audio input devices-------")
        for i in range (numdevices):
                if self.audio_controller.get_device_info_by_host_api_device_index(0,i).get('maxInputChannels') > 0:
                    device_name = self.audio_controller.get_device_info_by_host_api_device_index(0,i).get('name')
                    print(f"Input Device id {i} - {device_name}")
        print("-------------------------------------------")

    def __enter__(self):
        self.stream = self.audio_controller.open(input_device_index=self.device_idx,
                                                 format=self.format,
                                                 channels=self.channels,
                                                 rate=self.sampling_rate,
                                                 input=True,
                                                 frames_per_buffer=self.frames_per_buffer)
        return self

    def __exit__(self, *args, **kwargs):
        if self.stream is not None:
            self.stream.stop_stream()
            self.stream.close()
        self.audio_controller.terminate()

    def __iter__(self):
        if self.stream is None or self.audio_controller is None:
            return
        else:
            for i in range(self.frames_per_buffer):
                data = self.stream.read(self.frames_per_buffer)
                count = len(data) / 2
                fmt = "<H"
                data = np.array(list(struct.iter_unpack(fmt, data)))
                yield self.feature_extractor(data)


if __name__ == "__main__":
    from synethesia.network.feature_creators import logfbank_features
    with AudioRecorder(feature_extractor=logfbank_features) as rec:
        for audio in rec:
            print(audio)
