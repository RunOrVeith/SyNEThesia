import tensorflow as tf

from ops import *
from interfaces import Model


class SynethesiaModel(Model):

    def __init__(self, feature_dim):
        # TODO make image size a parameter
        self.feature_dim = feature_dim

        self.sound_feature = None
        self.reproduced_sound = None
        self.base_img = None
        self.generated_img = None

        super(SynethesiaModel, self).__init__()

    def initialize(self):
        self._build_model()

    def data_input(self):
        return self.sound_feature

    def data_output(self):
        return tf.tuple(tensors=[self.generated_img, self.reproduced_sound])

    def optimizer(self):
        return self._optimizer

    def learning_rate(self):
        return self._learning_rate

    def training_summary(self):
        return self._summary_op

    def global_step(self):
        return self._global_step

    def _img_from_sound(self, sound_feature, size=(1024, 512)):
        feature = self._feature_to_tensor(sound_feature=sound_feature, size=size)
        base_img = self._load_base_image(size=size)
        assert base_img.get_shape()[1:3] == feature.get_shape()[1:3], "Rows, Cols do not match"
        img_and_sound = tf.concat([base_img, feature], axis=-1, name="generator_input")
        return img_and_sound, base_img

    def _feature_to_tensor(self, sound_feature, size):
        tile_size = (1, size[0] * size[1])
        tensorized = tf.tile(sound_feature, tile_size)
        tensorized = tf.reshape(tensor=tensorized, shape=(-1, *size, self.feature_dim))
        return tensorized

    def _load_base_image(self, size):
        return tf.placeholder(dtype=tf.float32, shape=[None, *size, 3])

    def _build_model(self):

        with tf.variable_scope("synethesia"):
            self.sound_feature = tf.placeholder(dtype=tf.float32, shape=[None, self.feature_dim],
                                           name="feature_input")
            img_and_sound, self.base_img = self._img_from_sound(sound_feature=self.sound_feature)
            self.generated_img = self._build_encoder(x=img_and_sound)
            self.reproduced_sound = self._build_decoder(from_img=self.generated_img)
            assert self.reproduced_sound.get_shape()[1:] == self.sound_feature.get_shape()[1:]

            loss = self._build_loss(generated_img=self.generated_img,
                                    real_sound=self.sound_feature,
                                    generated_sound=self.reproduced_sound)
            self._global_step, self._learning_rate, self._optimizer = self._build_optimizer(loss=loss)

            self._summary_op = self._build_summary()

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

            _total_loss = lambda_reconstruct * reconstruction_loss + lambda_color * color_loss
            total_loss = tf.identity(_total_loss, name="loss")
            tf.losses.add_loss(total_loss)

        return total_loss

    def _build_optimizer(self, loss, decay_rate=1., decay_steps=100000):
        with tf.variable_scope("optimizer"):
            global_step = tf.get_variable(name="global_step", trainable=False, shape=[],
                                          initializer=tf.constant_initializer(0),
                                          dtype=tf.int64)
            learning_rate = tf.placeholder(dtype=tf.float32, shape=[],
                                           name="learning_rate")
            decayed_learning_rate = tf.train.exponential_decay(learning_rate=learning_rate,
                                                               global_step=global_step,
                                                               decay_steps=decay_steps, decay_rate=decay_rate,
                                                               staircase=False, name="rate_decay")
            optimizer = tf.train.AdamOptimizer(decayed_learning_rate).minimize(loss, global_step=global_step,
                                                                               name="optimizer")

        return global_step, learning_rate, optimizer


    def _build_summary(self):
        for loss in tf.losses.get_losses():
            tf.summary.scalar(name=loss.op.name, tensor=loss)

        return tf.summary.merge_all()



if __name__ == "__main__":
    net = SynethesiaModel(feature_dim=64)
    net.initialize()
