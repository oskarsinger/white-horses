import numpy as np

class ConstantLoader:

    def __init__(self, c):

        self.c = np.array([c])[:,np.newaxis]
        self.num_rounds = 0

    def get_data(self):

        self.num_rounds += 1

        return self.c

    def name(self):

        return 'ConstantLoader'

    def cols(self):

        return 1

    def rows(self):

        return self.num_rounds
