import tensorflow as tf
import numpy as np
from partygan.ops import *

class NeuralEqualizer(object):

    def __init__(self, batch_size=1, latent_dim=32, color_dim=1, scale=8.0, net_size=32):

        self.batch_size = batch_size
        self.net_size = net_size
        x_dim = 256
        y_dim = 256
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.scale = scale
        self.color_dim = color_dim
        self.latent_dim = latent_dim

        self.batch = tf.placeholder(tf.float32, [batch_size, x_dim, y_dim, color_dim])

        n_points = self.x_dim * self.y_dim
        self.n_points = n_points

        self.x_vec, self.y_vec,  self.r_vec = self.__coordinates(x_dim, y_dim, scale)

        self.latent = tf.placeholder(tf.float32, [self.batch_size, self.latent_dim])
        self.x = tf.placeholder(tf.float32, [self.batch_size, None, 1])
        self.y = tf.placeholder(tf.float32, [self.batch_size, None, 1])
        self.r = tf.placeholder(tf.float32, [self.batch_size, None, 1])

        self.G = self.generator(x_dim=self.x_dim, y_dim=self.y_dim)


    def init(self, sess):
        init = tf.global_variables_initializer()
        sess.run(init)

    def __coordinates(self, x_dim, y_dim, scale):
        n_points = x_dim * y_dim
        x_range = scale * (np.arange(x_dim) - (x_dim - 1) / 2.0) / (x_dim - 1)/ 0.5
        y_range = scale * (np.arange(y_dim) - (y_dim - 1) / 2.0) / (y_dim - 1)/ 0.5
        x_mat = np.matmul(np.ones((y_dim, 1)), x_range.reshape((1, x_dim)))
        y_mat = np.matmul(y_range.reshape((y_dim, 1)), np.ones((1, x_dim)))
        r_mat = np.sqrt(x_mat * x_mat + y_mat * y_mat)
        x_mat = np.tile(x_mat.flatten(), self.batch_size).reshape(self.batch_size, n_points, 1)
        y_mat = np.tile(y_mat.flatten(), self.batch_size).reshape(self.batch_size, n_points, 1)
        r_mat = np.tile(r_mat.flatten(), self.batch_size).reshape(self.batch_size, n_points, 1)
        return x_mat, y_mat, r_mat

    def generator(self, x_dim, y_dim, reuse=False):

        net_size = self.net_size
        n_points = x_dim * y_dim
        z_scaled = tf.reshape(self.latent, [self.batch_size, 1, self.latent_dim]) * tf.ones([n_points, 1], dtype=tf.float32) * self.scale
        z_unroll = tf.reshape(z_scaled, [self.batch_size * n_points, self.latent_dim])
        x_unroll = tf.reshape(self.x, [self.batch_size * n_points, 1])
        y_unroll = tf.reshape(self.y, [self.batch_size * n_points, 1])
        r_unroll = tf.reshape(self.r, [self.batch_size * n_points, 1])

        U = fully_connected(z_unroll, net_size, 'g_0_z') + \
            fully_connected(x_unroll, net_size, 'g_0_x', with_bias=False) + \
            fully_connected(y_unroll, net_size, 'g_0_y', with_bias=False) + \
            fully_connected(z_unroll, net_size, 'g_0_r', with_bias=False)

        H = tf.nn.tanh(U)
        for i in range(3):
            H = H + tf.nn.tanh(fully_connected(H, net_size, f'g_tanh_{i}'))

        output = tf.sigmoid(fully_connected(H, self.color_dim, 'g_final'))

        result = tf.reshape(output, [self.batch_size, y_dim, x_dim, self.color_dim])
        return result

    def generate(self, sess, x_dim, y_dim, scale, latent=None):
        if latent is None:
            latent = np.random.uniform(-1.0, 1.0, size=(self.batch_size, self.latent_dim)).astype(np.float32)

        x_vec, y_vec, r_vec = self.__coordinates(x_dim, y_dim, scale=scale)
        image = sess.run(self.G, feed_dict={self.latent: latent, self.x: x_vec, self.y: y_vec, self.r: r_vec})
        return image
