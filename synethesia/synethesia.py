import numpy as np
from pathlib import Path
from PIL import Image

from .synethesia_model import SynethesiaModel
from .model_interactor import TrainingSession, InferenceSession
from .interfaces import Trainable, Inferable
from .feature_creators import logfbank_features
from .audio_chunk_loader import StaticSongLoader


def random_start_img(img_size, batch_size):
    generation_shape = (batch_size, *img_size, 3)
    img = np.random.uniform(size=generation_shape)
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

    def __init__(self, song_files, img_size=(256, 128), batch_size=32):
        self.img_size = img_size
        self.batch_size = batch_size
        self.song_files = self._song_files_to_list(song_files=song_files)
        self.feature_extractor = logfbank_features

    def _song_files_to_list(self, song_files):
        if not isinstance(song_files, (list, tuple)):
            # TODO implement reading song  names from file
            raise NotImplemented("Files containg data not supported right now")
        return song_files

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

    def infer_and_stream(self, model_name, data_provider):
        # TODO implement streaming inference
        raise NotImplemented("Streaming based inference not implemented.")
