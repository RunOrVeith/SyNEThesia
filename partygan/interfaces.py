import abc

import tensorflow as tf

class Model(object, metaclass=abc.ABCMeta):

    def __init__(self):
        self._global_step = None
        self._learning_rate = None
        self._optimizer = None
        self._summary_op = None

    @abc.abstractmethod
    def initialize(self, graph):
        pass

    @property
    @abc.abstractmethod
    def data_input(self):
        pass

    @property
    @abc.abstractmethod
    def data_output(self):
        pass

    @property
    def optimizer(self):
        pass

    @property
    @abc.abstractmethod
    def learning_rate(self):
        pass

    @property
    def training_summary(self):
        return tf.no_op(name="summary_dummy")

    @property
    def global_step(self):
        pass
