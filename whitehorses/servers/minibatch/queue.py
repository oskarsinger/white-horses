import numpy as np

class Batch2Queue:

    def __init__(self, 
        batch_size, 
        data_loader):

        self.batch_size = batch_size
        self.dl = data_loader
        self.randomize = randomize

        self.data = self.dl.get_data()

    def get_data(self):


