import numpy as np

from whitehorses.loaders.simple import BernoulliLoader as BL
from whitehorses.loaders.simple import GaussianLoader as GL

class BernoulliGaussianCCALUPILoader:

    def __init__(self,
        W_o=None, W_p=None,
        mu_o=None, mu_p=None, 
        mu_zo=None, mu_zp=None,
        Psi_o=None, Psi_p=None,
        p=0.5,
        k=5
        fraction_flipped=0.2,
        bias=False):

        self.data = None

    def get_data(self):

        if self.data is None:
            self._set_data()

        return self.data

    def _set_data(self):

        y = BL(n, 1, p=self.p)
