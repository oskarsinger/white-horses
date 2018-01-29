import numpy as np

class BernoulliLoader:

    def __init__(self, n, m, p=0.5, lazy=True):

        self.n = n
        self.m = m
        self.p = p
        self.lazy = lazy

        self.data = None

        if not self.lazy:
            self._set_data()

    def get_data(self):

        if self.data is None:
            self._set_data()

        return self.data

    def _set_data(self):

        init = np.random.binomial(1, self.p, size=self.n*self.m)

        self.data = init.reshape((self.n, self.m))

    def rows(self):

        return self.n

    def cols(self):

        return self.m
