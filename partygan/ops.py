import tensorflow as tf
import numpy as np

def linear(input_, output_size, scope=None, stddev=1.0, bias_start=0.0, with_w=False):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size],
            initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias

def fully_connected(input_, output_size, scope=None, stddev=1.0, with_bias = True):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope or "FC"):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
            tf.random_normal_initializer(stddev=stddev))

        result = tf.matmul(input_, matrix)

        if with_bias:
            bias = tf.get_variable("bias", [1, output_size],
                initializer=tf.random_normal_initializer(stddev=stddev))
            result += bias*tf.ones([shape[0], 1], dtype=tf.float32)

        return result

def lrelu(x, leak=0.2, name="lrelu"):
    assert leak < 1 and leak > 0
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)


def conv2d(x, output_dim, kernel_size=(5, 5), stride_diff=(2, 2), stddev=0.02, bias_start=0., scope=None):
    with tf.variable_scope(scope or "conv2d"):
        w = tf.get_variable("w", list(kernel_size) + [x.get_shape()[-1]] + [output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(x, w, strides=(1,) + stride_diff + (1,), padding="SAME")
        biases = tf.get_variable("biases", [output_dim], initializer=tf.constant_initializer(bias_start))
        conv = tf.nn.bias_add(conv, biases)

        return conv


def conv2d_transpose(x, output_dim, kernel_size=(5, 5), stride_diff=(2, 2), stddev=0.02, scope=None, with_w=False):
    with tf.variable_scope(scope or "conv2d_transpose"):
        w = tf.get_variable("w", list(kernel_size) + [output_dim[-1]] + [x.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))

        deconv = tf.nn.conv2d_transpose(x, w, output_shape=output_dim, strides=(1,) + stride_diff + (1,))
        biases = tf.get_variable("biases", [output_dim[-1]], initializer=tf.constant_initializer(0.))
        deconv = tf.nn.bias_add(deconv, biases)

        if with_w:
            return deconv, w, biases
        else:
            return deconv

class BatchNorm(object):

    def __init__(self, epsilon=1e-5, momentum=0.9, scope=None):
        with tf.variable_scope(scope or "batch_norm") as scope:
            self.epsilon = epsilon
            self.momentum = momentum
            self.name = scope.name

    def __call__(self, x, train):
        return tf.contrib.layers.batch_norm(x, decay=self.momentum, updates_collections=None, epsilon=self.epsilon,
                                            center=True, scale=True, is_training=train, scope=self.name)
