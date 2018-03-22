import tensorflow as tf

from .ops import *
from .interfaces import Model


class SynethesiaModel(Model):

    def __init__(self, feature_dim, img_size=(256, 128), num_residual=3):

        self.feature_dim = feature_dim
        self.img_size = img_size

        self._num_3x3 = int(np.floor(np.log2(self.img_size[0])) - 4)
        self._channels = 2 ** self._num_3x3
        self._num_residual = num_residual
        self._num_transpose = self._num_3x3

        self.sound_feature = None
        self.reproduced_sound = None
        self.base_img = None
        self.generated_img = None

        super().__init__()

    def initialize(self):
        self._build_model()

    @property
    def data_input(self):
        return self.sound_feature

    @property
    def data_output(self):
        return tf.tuple(tensors=[self.generated_img, self.reproduced_sound])

    @property
    def optimizer(self):
        return self._optimizer

    @property
    def learning_rate(self):
        return self._learning_rate

    @property
    def training_summary(self):
        return self._summary_op

    @property
    def global_step(self):
        return self._global_step

    def _img_from_sound(self, sound_feature):
        feature = self._feature_to_tensor(sound_feature=sound_feature)
        base_img = self._load_base_image()
        assert base_img.get_shape()[1:3] == feature.get_shape()[1:3], "Rows, Cols do not match"
        img_and_sound = tf.concat([base_img, feature], axis=-1, name="generator_input")
        return img_and_sound, base_img

    def _feature_to_tensor(self, sound_feature):
        tile_size = (1, self.img_size[0] * self.img_size[1])
        tensorized = tf.tile(sound_feature, tile_size)
        tensorized = tf.reshape(tensor=tensorized, shape=(-1, *self.img_size, self.feature_dim))
        return tensorized

    def _load_base_image(self):
        return tf.placeholder(dtype=tf.float32, shape=[None, *self.img_size, 3])

    def _build_model(self):
        # TODO dont ignore base image
        # TODO compare difference to previous slice
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
        channels = self._channels
        with tf.variable_scope("encoder"):

            x = conv2d(x=x, output_channels=channels, kernel=(7, 7), stride=1, scope=f"7x7_{channels}")

            for i in range(self._num_3x3,  2 * self._num_3x3):
                channels *= 2
                x = conv2d(x=x, output_channels=channels,
                           kernel=(3, 3), stride=2,
                           use_batchnorm=True,
                           activation=tf.nn.relu,
                           scope=f"3x3_{channels}")

            x = residual_block(x=x, output_channels=channels, use_batchnorm=True,
                               activation=tf.nn.relu, scope="residual_0")
            for i in range(self._num_residual - 1):
                x = residual_block(x=x, output_channels=channels, scope=f"residual_{i+1}")

            channels //= 2
            x = conv2d_transposed(x=x, output_channels=channels, kernel=(3, 3), stride=2,
                                  scope=f"transposed_{channels}")
            for i in range(self._num_transpose - 1):
                channels //= 2
                x = conv2d_transposed(x=x, output_channels=channels,
                                      kernel=(3, 3), stride=2,
                                      use_batchnorm=True,
                                      activation=tf.nn.relu,
                                      scope=f"transposed_{channels}")

            x = conv2d(x=x, output_channels=3, kernel=(7, 7), stride=1, activation=tf.nn.relu, scope="7x7_3")
            x = tf.nn.sigmoid(x, name="encoder")  # Scale pixels to [0,1]

        return x

    def _build_decoder(self, from_img):

        with tf.variable_scope("decoder"):
            y = conv2d(x=from_img, output_channels=self._channels, kernel=(7, 7), stride=1, scope="7x7_64")

            y = residual_block(x=y, output_channels=self._channels, activation=tf.nn.relu, scope="residual_0")
            for i in range(self._num_residual - 1):
                y = residual_block(x=y, output_channels=self._channels, scope=f"residual_{i+1}")

            y = conv2d(x=y, output_channels=self.feature_dim, kernel=(3, 3), stride=1, scope=f"3x3_{self.feature_dim}")
            y = tf.reduce_mean(y, [1, 2], name="decoder")
            # No activation because we don't want any value limits
        return y

    def _build_loss(self, generated_img, real_sound, generated_sound,
                    lambda_reconstruct=10., lambda_color=1.0, lambda_colorfulness=1.0, lambda_noise=1.):

        with tf.variable_scope("loss"):

            sound_reconstruction_loss = self._add_sound_reconstruction_loss(real_sound=real_sound,
                                                                            generated_sound=generated_sound)
            color_loss = self._add_color_loss(generated_img)
            colorfulness_loss = self._add_colorfulness_loss(generated_img)
            noise_loss = self._add_noise_loss(generated_img)

            _total_loss = (lambda_reconstruct * sound_reconstruction_loss +
                           lambda_color * color_loss +
                           lambda_colorfulness * colorfulness_loss +
                           lambda_noise * noise_loss)
            total_loss = tf.identity(_total_loss, name="total_loss")
            tf.losses.add_loss(total_loss)

        return total_loss

    def _add_sound_reconstruction_loss(self, real_sound, generated_sound):
        sound_reconstruction_loss = tf.losses.mean_squared_error(labels=real_sound, predictions=generated_sound,
                                                                 scope="sound_reconstruction_loss")
        return sound_reconstruction_loss

    def _add_color_loss(self, generated_img):
        mean_global_color, _ = tf.nn.moments(generated_img, axes=[1, 2], name="global_color", keep_dims=True)
        _, colorfulness = tf.nn.moments(tf.abs(generated_img - mean_global_color), axes=[-1], name="colorfulness")
        _color_loss = - tf.reduce_sum(colorfulness)
        color_loss = _color_loss
        color_loss = tf.identity(color_loss, name="color_loss")
        tf.losses.add_loss(color_loss)
        return color_loss

    def _add_colorfulness_loss(self, generated_img, num_colors=9):
        binned_values = tf.reshape(tf.floor(generated_img * (num_colors - 1)), [-1])
        binned_values = tf.cast(binned_values, tf.int32)
        ones = tf.ones_like(binned_values, dtype=tf.int32)
        histogram = tf.unsorted_segment_sum(ones, binned_values, num_colors)
        colorfulness_loss = tf.cast(- tf.reduce_max(histogram), tf.float32, name="colorfulness_loss")
        tf.losses.add_loss(colorfulness_loss)
        return colorfulness_loss

    def _add_noise_loss(self, generated_img):
        _noise_loss = tf.reduce_sum(tf.image.total_variation(images=generated_img))
        noise_loss = tf.divide(_noise_loss, tf.cast(tf.size(generated_img), tf.float32), name="noise_loss")
        tf.losses.add_loss(noise_loss)
        return noise_loss

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
