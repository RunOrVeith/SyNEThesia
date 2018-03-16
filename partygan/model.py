from pathlib import Path
from collections import defaultdict

import tensorflow as tf
import numpy as np

from partygan.ops import *


class SynethesiaModel(object):

    def __init__(self, feature_dim):
        self.feature_dim = feature_dim

        (self.sound_feature, self.reproduced_sound,
         self.base_img, self.generated_img,
         self.loss, self.merged_summary) = self._build_model()

    def _placeholder(self, dtype, shape=None, name=None):
        _placeholder = tf.placeholder(dtype=dtype, shape=shape, name=name)
        tf.add_to_collection(name="placeholders", value=_placeholder)
        return _placeholder

    def _img_from_sound(self, sound_feature, size=(1024, 512)):
        feature = self._feature_to_tensor(sound_feature=sound_feature, size=size)
        base_img = self._load_base_image(size=size)
        assert base_img.get_shape()[1:3] == feature.get_shape()[1:3], "Rows, Cols do not match"
        x = tf.concat([base_img, feature], axis=-1, name="generator_input")
        return x, base_img

    def _feature_to_tensor(self, sound_feature, size):
        tile_size = (1, size[0] * size[1])
        tensorized = tf.tile(sound_feature, tile_size)
        tensorized = tf.reshape(tensor=tensorized, shape=(-1, *size, self.feature_dim))
        return tensorized

    def _load_base_image(self, size):
        return self._placeholder(dtype=np.float32, shape=[None, *size, 3])

    def _build_model(self):

        with tf.variable_scope("synethesia"):
            sound_feature = self._placeholder(dtype=tf.float32, shape=[None, self.feature_dim],
                                              name="feature_input")
            x, base_img = self._img_from_sound(sound_feature=sound_feature)
            generated_img = self._build_encoder(x=x)
            reproduced_sound = self._build_decoder(from_img=generated_img)
            assert reproduced_sound.get_shape()[1:] == sound_feature.get_shape()[1:]

            loss = self._build_loss(generated_img=generated_img,
                                    real_sound=sound_feature,
                                    generated_sound=reproduced_sound)
            summary = self._build_summary()
        return sound_feature, reproduced_sound, base_img, generated_img, loss, summary

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

            x = conv2d(x=x, output_channels=3, kernel=(7, 7), stride=1, activation=tf.nn.relu, scope="7x7_3")
            x = tf.nn.sigmoid(x, name="encoder")  # Scale pixels to [0,1]
        return x

    def _build_decoder(self, from_img):
        num_residual = 9
        with tf.variable_scope("decoder"):
            y = conv2d(x=from_img, output_channels=64, kernel=(7, 7), stride=1, scope="7x7_64")

        y = residual_block(x=y, output_channels=64, activation=tf.nn.relu, scope="residual_0")
        for i in range(num_residual):
            y = residual_block(x=y, output_channels=64, scope=f"residual_{i+1}")

        y = conv2d(x=y, output_channels=self.feature_dim, kernel=(3, 3), stride=1, scope=f"3x3_{self.feature_dim}")
        y = tf.reduce_mean(y, [1, 2], name="decoder")
        # No activation because we don't want any value limits
        return y

    def _build_loss(self, generated_img, real_sound, generated_sound, lambda_reconstruct=0.9, lambda_color=0.1):

        with tf.variable_scope("loss"):
            reconstruction_loss = tf.losses.huber_loss(labels=real_sound, predictions=generated_sound,
                                                       delta=1.0, scope="huber_loss")
            _, colorfulness = tf.nn.moments(generated_img, axes=[-1], name="colorfulness")
            _color_loss = - tf.reduce_sum(colorfulness) / tf.cast(tf.size(colorfulness), tf.float32)
            color_loss = tf.identity(_color_loss, name="color_loss")
            tf.losses.add_loss(color_loss)

            total_loss = lambda_reconstruct * reconstruction_loss + lambda_color * color_loss
            tf.losses.add_loss(total_loss)

            return tf.identity(total_loss, name="loss")

    def _build_summary(self):
        for loss in tf.losses.get_losses():
            tf.summary.scalar(name=loss.op.name, tensor=loss)

        return tf.summary.merge_all()



if __name__ == "__main__":
    net = SynethesiaModel(feature_dim=64)
