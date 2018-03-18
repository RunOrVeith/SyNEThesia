import abc

import numpy as np

from model import SynethesiaModel
from model_interactor import TrainingSession, InferenceSession


class Trainable(object, metaclass=abc.ABCMeta):

    def __init__(self, model):
        self.training_session = TrainingSession(model=model, generate_train_dict=self.generate_train_dict)

    @abc.abstractmethod
    def generate_train_dict(self, learning_rate, input_features, batch_size):
        pass


class Inferable(object, metaclass=abc.ABCMeta):

    def __init__(self, model):
        self.inferece_session = InferenceSession(model=model, generate_inference_dict=self.generate_inference_dict)

    @abc.abstractmethod
    def generate_inference_dict(self, input_features, batch_size):
        pass


class Synethesia(Trainable, Inferable):

    def __init__(self, feature_dim, img_size=(1024, 512)):
        self.img_size = img_size
        self.model = SynethesiaModel(feature_dim=feature_dim)
        Trainable.__init__(self, model=self.model)
        Inferable.__init__(self, model=self.model)

    def generate_train_dict(self, learning_rate, input_features, batch_size):
        feed_dict = {self.model.learning_rate: learning_rate,
                     self.model.data_input: input_features,
                     self.model.base_img: self.random_start_img(batch_size=batch_size)}
        return feed_dict

    def generate_inference_dict(self, input_features, batch_size):
        feed_dict = {self.model.data_input: input_features,
                     self.model.base_img: self.random_start_img(batch_size=batch_size)}
        return feed_dict

    def random_start_img(self, batch_size):
        generation_shape = (batch_size, *self.img_size, 3)
        img = np.random.uniform(size=generation_shape)
        return img


if __name__ == "__main__":
    from audio_chunk_loader import StaticSongLoader
    synethesia = Synethesia(feature_dim=64)
    train_loader = StaticSongLoader(song_files=["/home/veith/Projects/PartyGAN/data/Bearded Skull - 420 [Hip Hop Instrumental]/audio/soundtrack.mp3"],
                                    batch_size=1, load_n_songs_at_once=1)
    synethesia.training_session.train(model_name="overfit_bearded_skull", data_provider=train_loader)
