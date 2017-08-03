import numpy as np

from drrobert.stats import get_zm_uv

class PlainQueue:

    def __init__(self, 
        batch_size, 
        center=False):

        self.batch_size = batch_size
        self.center = center

        self.batch = None
        self.sum = None

    def add_data(self, data):

        data = np.copy(data)

        if self.batch is None:
            self.sum = data
            self.batch = data
        elif self.batch.shape[0] < self.batch_size:
            self.sum += data
            self.batch = np.vstack([
                self.batch, 
                data])
        else:
            self.sum += data - self.batch[0,:]
            self.batch = np.vstack([
                self.batch[1:,:],
                data])

    def get_mean(self):

        return self.sum / self.batch.shape[0]

    def get_batch(self):

        batch = np.copy(self.batch)

        if self.center:
            batch -= self.get_mean()

        return batch
