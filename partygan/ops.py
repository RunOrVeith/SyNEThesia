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

def rfft(_in, name='rfft', graph=tf.get_default_graph()):
    with graph.as_default():
        with tf.name_scope(name):
            cast = tf.complex(tf.cast(_in, tf.float32, name='cast_to_float32'), tf.constant(0.0, dtype=tf.float32), name='cast_to_complex')
            fftOp = tf.fft(cast, name='fft')
            half, _ = tf.split(0, 2, fftOp, name='split')
            double = tf.mul(tf.constant(2.0, dtype=tf.complex64), half)
            return double
