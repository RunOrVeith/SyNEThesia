import numpy as np
import random

class BatchCreator(object):

    def __init__(self, iterable, batch_size):
        self._iterator_source = iterable
        self.iterator = None
        self.batch_size = batch_size

        self.reset()

    def __iter__(self):
        return self

    def reset_and_shuffle(self):
        random.shuffle(self._iterator_source)
        self.reset()

    def reset(self):
        self.iterator = iter(self._iterator_source)

    def __iter__(self):
        return self

    def __next__(self):
        instances = []
        try:
            for _ in range(self.batch_size):
                instances.append(next(self.iterator))
        except StopIteration:
            # Number of elements in iterator is not a multuple of batch_size,
            # just batch the remaining instances
            if len(instances) == 0:
                raise StopIteration()

        return np.stack(instances, axis=0)
