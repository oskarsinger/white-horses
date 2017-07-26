import numpy as np

from drrobert.stats import get_zm_uv

class ExponentialDownWeightedQueue:

    def __init__(self, batch_size, alpha=0.99):

        self.batch_size = batch_size
        self.alpha = alpha

        self.data_sum = None
        self.alpha_sum = 0
        self.batch = None
        self.num_rounds = 0

    def add_data(self, data):

        data = np.copy(data)
        self.alpha_sum += self.alpha**(self.num_rounds)

        if self.num_rounds == 0:
            self.data_sum = data
            self.batch = data 
        else:
            self.data_sum = self.alpha * self.data_sum + data
            new_data = self.data_sum / self.alpha_sum

            if self.batch.shape[0] < self.batch_size:
                self.batch = np.vstack([
                    self.batch,
                    new_data])
            else:
                self.batch = np.vstack([
                    self.batch[1:,:],
                    new_data])

    def get_batch(self):

        return get_zm_uv(self.batch)
