import numpy as np
from pathlib import Path
from PIL import Image

from synethesia.network import (SynethesiaModel, logfbank_features, fft_features, StaticSongLoader,
                                LiveViewer, VideoCreator, AudioRecorder)
from synethesia.framework import TrainingSession, InferenceSession, Trainable, Inferable


def random_start_img(img_size, batch_size, num_channels=3, num_ones_offset=None):
    zeros = np.zeros((*img_size, batch_size * num_channels))
    num_ones_offset = np.random.choice([1, 0], size=batch_size * num_channels)
    zeros += num_ones_offset * np.random.random(size=batch_size * num_channels)
    img = zeros.reshape((batch_size, *img_size, num_channels))
    return img


class SynethesiaTrainer(Trainable):

    def __init__(self, img_size, model):
        self.img_size = img_size
        self.model = model
        super().__init__()

    def generate_train_dict(self, input_features):
        batch_size = input_features.shape[0]
        feed_dict = {self.model.data_input: input_features,
                     self.model.base_img: random_start_img(self.img_size, batch_size=batch_size)}
        return feed_dict


class SynethesiaInferer(Inferable):

    def __init__(self, img_size, model):
        self.img_size = img_size
        self.model = model
        super().__init__()

    def generate_inference_dict(self, input_features):
        batch_size = input_features.shape[0]
        feed_dict = {self.model.data_input: input_features,
                     self.model.base_img: random_start_img(self.img_size, batch_size=batch_size)}
        return feed_dict


class Synethesia(object):

    def __init__(self, song_files, img_size=(256, 128), batch_size=32, feature_extractor=logfbank_features):
        self.img_size = img_size
        self.batch_size = batch_size
        self.song_files = self._song_files_to_list(song_files=song_files)
        self.feature_extractor = feature_extractor

    def _song_files_to_list(self, song_files):
        if not isinstance(song_files, (list, tuple)):
            pth = Path(song_files).resolve()

            print(f"Reading songs from {str(pth)}.")
            if not pth.exists():
                raise ValueError(f"Could not find {str(song_files)}")

            if pth.is_dir():
                contents = [str(content) for content in pth.iterdir() if content.is_file() and content.suffix == ".mp3"]
            elif pth.is_file():
                if pth.suffix == ".mp3":
                    contents =  str(pth),
                else:
                    contents = [content.strip() for content in pth.read_text().splitlines()]

        else:
            contents = song_files

        print(f"Received {len(contents)} sound files.")
        return contents

    def train(self, model_name, learning_rate=0.0001, songs_at_once=3):
        train_loader = StaticSongLoader(song_files=self.song_files, feature_extractor=self.feature_extractor,
                                        batch_size=self.batch_size, load_n_songs_at_once=songs_at_once,
                                        to_infinity=True, allow_shuffle=True)
        model = SynethesiaModel(feature_dim=train_loader.feature_dim, img_size=self.img_size)
        train_session = TrainingSession(model=model,
                                        learning_rate=learning_rate,
                                        trainable=SynethesiaTrainer(img_size=self.img_size, model=model))
        print("Starting to train...")
        train_session.train(model_name=model_name, data_provider=train_loader)

    def _infer(self, model_name, data_provider):
        model = SynethesiaModel(feature_dim=data_provider.feature_dim, img_size=self.img_size)
        inference_session = InferenceSession(model=model, inferable=SynethesiaInferer(img_size=self.img_size, model=model))
        yield from inference_session.infer(model_name=model_name, data_provider=data_provider)

    def infer_and_store(self, model_name, target_dir="/tmp"):
        if len(self.song_files) > 1:
            print("Trying to infer more than one song! Will only infer the first one. (Will be fixed).")
            # TODO Make StaticSongLoader indicate end of song to allow inference of multiple songs at once

        infer_loader = StaticSongLoader(song_files=(self.song_files[0],), feature_extractor=self.feature_extractor,
                                        batch_size=self.batch_size, load_n_songs_at_once=1,
                                        to_infinity=False, allow_shuffle=False)

        # Assume data has been downloaded with the provided script
        song_name = Path(self.song_files[0]).parts[-3]
        _target_dir = Path(target_dir) / song_name
        _target_dir.mkdir(parents=True, exist_ok=True)

        img_id = 0
        print("Starting inference...")
        for i, (imgs, sounds) in enumerate(self._infer(model_name=model_name, data_provider=infer_loader)):
            for j, img in enumerate(imgs):
                img_id += 1
                Image.fromarray((img * 255).astype(np.uint8)).save(str(_target_dir / f"{img_id}.png"))

        video_creator = VideoCreator()
        video_creator(png_folder=_target_dir, mp3_file=self.song_files[0])

    def infer_and_stream(self, model_name, approx_fps=24, border_color="black"):

        with AudioRecorder(feature_extractor=self.feature_extractor) as infer_loader:
            generator = self._infer(model_name=model_name, data_provider=infer_loader)

            def yield_single_img():
                for imgs, sounds in generator:
                    for img in imgs:
                        yield img

            print("Starting streaming inference...")
            viewer = LiveViewer(approx_fps=approx_fps, border_color=border_color)
            viewer.toggle_fullscreen()
            viewer.display(image_generator=yield_single_img)
