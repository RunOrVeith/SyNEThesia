import numpy as np

from model import SynethesiaModel
from model_interactor import TrainingSession, InferenceSession


class Synethesia(object):

    def __init__(self, feature_dim, img_size=(1024, 512)):
        self.img_size = img_size
        self.model = SynethesiaModel(feature_dim=feature_dim)
        self.training_session = TrainingSession(model=self.model,
                                                generate_train_dict=self.generate_train_dict)
        self.inferece_session = InferenceSession(model=self.model,
                                                 generate_inference_dict=self.generate_inference_dict)

    def generate_train_dict(self, learning_rate, input_feature, batch_size):
        feed_dict = {self.model.learning_rate: learning_rate,
                     self.model.data_input: input_feature,
                     self.model.base_img: self.random_start_img(batch_size=batch_size)}
        return feed_dict

    def generate_inference_dict(self, input_feature, batch_size):
        feed_dict = {self.model.data_input: input_feature,
                     self.model.base_img: self.random_start_img(batch_size=batch_size)}
        return feed_dict

    def random_start_img(batch_size):
        generation_shape = (batch_size, *self.img_size, 3)
        img = np.random.uniform(size=generation_shape)
        return img


if __name__ == "__main__":
    synethesia = Synethesia(feature_dim=64)
