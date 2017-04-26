import numpy as np

from numbers import Number

class ConstantLoader:

    def __init__(self, c):

        if isinstance(c, Number):
            c = np.array([c])[:,np.newaxis]

        self.c = c
        self.num_rounds = 0

    def get_data(self):

        self.num_rounds += 1
        
        return self.c

    def name(self):

        return 'ConstantLoader'

    def cols(self):

        return self.c[0]

    def rows(self):

        return self.num_rounds
