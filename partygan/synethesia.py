import numpy as np

from model import SynethesiaModel


class Synethesia(Trainable):

    def __init__(self, feature_dim, img_size=(1024, 512)):
        self.img_size = img_size
        super(Trainable, self).__init__(SynethesiaModel(feature_dim=feature_dim))

    def generate_train_feed_dict(self, learning_rate, input_feature, batch_size):
        feed_dict = {self.model.learning_rate: learning_rate,
                     self.model.data_input: input_feature,
                     self.model.base_img: self.random_start_img(batch_size=batch_size)}
        return feed_dict

    def random_start_img(batch_size):
        generation_shape = (batch_size, *self.img_size, 3)
        img = np.random.uniform(size=generation_shape)
        return img
