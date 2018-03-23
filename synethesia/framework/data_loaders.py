import random

import numpy as np


class BatchCreator(object):

    def __init__(self, iterable, batch_size, allow_shuffle=False):
        self._iterator_source = iterable
        self.iterator = None
        self.batch_size = batch_size
        self.allow_shuffle = allow_shuffle
        self.reset()

    def __iter__(self):
        return self

    def reset(self):
        if self.allow_shuffle:
                    random.shuffle(self._iterator_source)
        self.iterator = iter(self._iterator_source)

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
