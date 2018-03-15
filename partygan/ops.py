import tensorflow as tf
from tensorflow.python.ops.random_ops import truncated_normal
import numpy as np


def conv2d(x, output_channels, scope=None, kernel=(5, 5), activation=None, stride=1):

    strides = [1, stride, stride, 1]
    in_dim = x.get_shape().as_list()[-1]

    if activation is not None:
        x = activation(x)

    with tf.variable_scope(scope or "conv2"):
        initializer = tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32)
        W = tf.get_variable(name="weight", shape=[*kernel, in_dim, output_channels],
                            dtype=tf.float32, initializer=initializer,
                            trainable=True)

        b = tf.get_variable(name="bias", shape=[output_channels],
                            dtype=tf.float32, initializer=tf.constant_initializer(0.),
                            trainable=True)

        x = tf.nn.conv2d(input=x, filter=W, strides=strides, padding='SAME',
                         use_cudnn_on_gpu=True,
                         data_format='NHWC')
        x = tf.add(x, b)

    return x


def conv2d_transposed(x, output_channels, scope=None, kernel=(4, 4), stride=2, activation=None):
    assert isinstance(output_channels, int)
    assert isinstance(stride, int)

    channels = max(output_channels, 2)
    strides = [1, stride, stride, 1]
    input_channels = x.get_shape()[-1].value

    with tf.variable_scope(scope or "conv2d_transposed"):

        if activation is not None:
            x = activation(x)

        input_shape = tf.shape(x)
        h = input_shape[1] * stride
        w = input_shape[2] * stride
        output_shape = tf.stack([input_shape[0], h, w, channels])

        bilinear_initializer = bilinear_weight_initializer(filter_size=kernel)

        W = tf.get_variable(name="weight", shape=[*kernel, channels, input_channels],
                            dtype=tf.float32, initializer=bilinear_initializer,
                            trainable=True)

        b = tf.get_variable(name="bias", shape=[output_channels],
                            dtype=tf.float32, initializer=tf.constant_initializer(0.),
                            trainable=True)

        x = tf.nn.conv2d_transpose(value=x, filter=W, output_shape=output_shape, strides=strides,
                                   padding='SAME', data_format='NHWC', name="conv2d_transposed")

    return tf.add(x, b)


def bilinear_weight_initializer(filter_size, add_noise=True):

    def _initializer(shape, dtype=tf.float32, partition_info=None):
        if shape:
            # second last dimension is input, last dimension is output
            fan_in = float(shape[-2]) if len(shape) > 1 else float(shape[-1])
            fan_out = float(shape[-1])
        else:
            fan_in = 1.0
            fan_out = 1.0

        # define weight matrix (set dtype always to float32)
        weights = np.zeros((filter_size[0], filter_size[1], int(fan_in), int(fan_out)), dtype=dtype.as_numpy_dtype())

        # get bilinear kernel
        bilinear = bilinear_filt(filter_size=filter_size)
        bilinear = bilinear / fan_out  # normalize by number of channels

        # set filter in weight matrix (also allow channel mixing)
        for i in range(weights.shape[2]):
            for j in range(weights.shape[3]):
                weights[:, :, i, j] = bilinear

        # add small noise for symmetry breaking
        if add_noise:
            # define standard deviation so that it is equal to 1/2 of the smallest weight entry
            std = np.min(bilinear) / 2
            noise = truncated_normal(shape=shape, mean=0.0,
                                                              stddev=std, seed=None, dtype=dtype)
            weights += noise

        return weights

    return _initializer

def bilinear_filt(filter_size=(4, 4)):
    assert isinstance(filter_size, (list, tuple)) and len(filter_size) == 2

    factor = [(size + 1) // 2 for size in filter_size]

    if filter_size[0] % 2 == 1:
        center_x = factor[0] - 1
    else:
        center_x = factor[0] - 0.5

    if filter_size[1] % 2 == 1:
        center_y = factor[1] - 1
    else:
        center_y = factor[1] - 0.5

    og = np.ogrid[:filter_size[0], :filter_size[1]]
    kernel = (1 - abs(og[0] - center_x) / factor[0]) * (1 - abs(og[1] - center_y) / factor[1])

    return kernel

def residual_block(x, output_channels, scope=None, activation=None):
    with tf.variable_scope(scope or "residual_block"):
        shortcut = x
        input_channels = x.get_shape()[-1]

        x = conv2d(x=x,
                   output_channels=output_channels,
                   scope='conv_1',
                   kernel=(3, 3),
                   stride=1,
                   activation=activation)

        x = conv2d(x=x,
                   output_channels=output_channels,
                   scope='conv_2',
                   kernel=(3, 3),
                   stride=1,
                   activation=activation)

        if input_channels != output_channels:
            shortcut = conv2d(x=shortcut,
                              output_channel=output_channels,
                              scope='conv_1x1',
                              kernel=(1, 1),
                              stride=1,
                              activation=None)

    return tf.add(shortcut, x)
