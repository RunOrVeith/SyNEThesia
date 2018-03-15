from pathlib import Path
from collections import defaultdict

import tensorflow as tf
import numpy as np

from partygan.ops import *


class TypeCollector(object):

    @staticmethod
    def placeholder(dtype, shape=None, name=None):
        _placeholder = tf.placeholder(dtype=dtype, shape=shape, name=name)
        tf.add_to_collection(name="placeholders", value=_placeholder)
        return _placeholder


class SyNetHesia(object):

    def __init__(self, feature_dim):
        self.model_name = "synethesia"
        self.feature_dim = feature_dim

        self.build_model()

    @property
    def model_name(self):
        return self._model_name

    @model_name.setter
    def model_name(self, model_name):
        self._model_name = model_name
        self.checkpoint_dir = Path("./checkpoints") / self.model_name
        self.logdir = Path("./logs") / self.model_name

    def _img_from_sound(self, sound_feature):
        feature = self._feature_to_tensor(sound_feature)
        base_img = self._load_base_image(feature)
        assert base_img.get_shape()[1:3] == feature.get_shape()[1:3], "Rows, Cols do not match"
        x = tf.concat([base_img, feature], axis=-1, name="generator_input")
        return x

    def _feature_to_tensor(self, sound_feature, size=(1024, 512)):
        tile_size = (1, size[0] * size[1])
        tensorized = tf.tile(sound_feature, tile_size)
        tensorized = tf.reshape(tensor=tensorized, shape=(-1, *size, self.feature_dim))
        return tensorized

    def _load_base_image(self, feature=None, size=(1024, 512)):
        batch_size = tf.shape(feature)[0]
        generation_shape = (batch_size, *size, 3)
        img = tf.random_uniform(shape=generation_shape, minval=0., maxval=1.,
                                dtype=tf.float32, name="random_start_img")
        return img

    def build_model(self):

        with tf.variable_scope("synethesia"):
            sound_feature = TypeCollector.placeholder(dtype=tf.float32, shape=[None, self.feature_dim],
                                                     name="feature_input")
            x = self._img_from_sound(sound_feature=sound_feature)
            generated_img = self._build_encoder(x=x)
            reproduced_sound = self._build_decoder(from_img=generated_img)
            assert reproduced_sound.get_shape()[1:] == sound_feature.get_shape()[1:]

    def _build_encoder(self, x):
        num_3x3 = 4
        num_residual = 9
        num_transpose = num_3x3
        with tf.variable_scope("encoder"):
            x = conv2d(x=x, output_channels=64, kernel=(7, 7), stride=1, scope="7x7_64")

            channels = 2 ** 6
            for i in range(7, 7 + num_3x3):
                channels *= 2
                x = conv2d(x=x, output_channels=channels,
                           kernel=(3, 3), stride=2,
                           activation=tf.nn.relu,
                           scope=f"3x3_{channels}")  # TODO add normalization

            assert channels == 1024

            x = residual_block(x=x, output_channels=channels,
                               activation=tf.nn.relu, scope="residual_0")  # TODO add normalization
            for i in range(num_residual - 1):
                x = residual_block(x=x, output_channels=channels, scope=f"residual_{i+1}")

            channels //= 2
            x = conv2d_transposed(x=x, output_channels=channels, kernel=(3, 3), stride=2,
                                  scope=f"transposed_{channels}")
            for i in range(num_transpose - 1):
                channels //= 2
                x = conv2d_transposed(x=x, output_channels=channels,
                                      kernel=(3, 3), stride=2,
                                      activation=tf.nn.relu,
                                      scope=f"transposed_{channels}") # TODO add normalization
            assert channels == 64
            x = conv2d(x=x, output_channels=3, kernel=(7, 7), stride=1, activation=tf.nn.relu, scope="7x7_3")
        return x

    def _build_decoder(self, from_img):
        num_residual = 9
        with tf.variable_scope("decoder"):
            y = conv2d(x=from_img, output_channels=64, kernel=(7, 7), stride=1, scope="7x7_64")

        y = residual_block(x=y, output_channels=64, activation=tf.nn.relu, scope="residual_0")
        for i in range(num_residual):
            y = residual_block(x=y, output_channels=64, scope=f"residual_{i+1}")

        y = conv2d(x=y, output_channels=self.feature_dim, kernel=(3, 3), stride=1, scope=f"3x3_{self.feature_dim}")
        y = tf.reduce_mean(y, [1, 2])
        y = tf.nn.relu(y, name="extracted_features")
        return y

if __name__ == "__main__":
    net = SyNetHesia(64)
