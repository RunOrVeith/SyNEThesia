import abc


class Model(object, metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def initialize(self, graph):
        pass
