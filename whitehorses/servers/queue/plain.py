import numpy as np

class PlainQueue:

    def __init__(self, batch_size):

        self.batch_size = batch_size

        self.batch = None

    def add_data(self, data):

        data = np.copy(data)
         
        if self.batch is None:
            self.batch = data
        elif self.batch.shape[0] < self.batch_size:
            self.batch = np.vstack([
                self.batch, 
                data])
        else:
            self.batch = np.vstack([
                self.batch[1:,:],
                data])

    def get_batch(self):

        return self.batch
