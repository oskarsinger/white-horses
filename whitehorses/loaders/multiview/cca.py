import numpy as np

from linal.svd import get_svd_power
from linal.random import get_sparse_normal

# TODO: make some util functions for generating special cases
# TODO: cite the Francis Bach and Fu 2016 papers

def get_Fu2016_data(num_data, k, ds, lazy=True, density=0.1):

    Z = get_sparse_normal(k, num_data, density=density)

    return [Fu2016Loader(Z, d, lazy=lazy, density=density)
            for d in ds]

def get_easy_SCCAPMLs(num_data, k, ds, lazy=True):

    Z = np.random.randn(k, num_data)
    Psi_inits = [np.random.randn(d * 2, d)
                 for d in ds]
    Psis = [np.dot(Pi.T, Pi) for Pi in Psi_inits]
    Ws = [np.random.randn(d, k) for d in ds]
    mus = [np.random.randn(d, 1) for d in ds]
    zipped = zip(
        Ws,
        Psis,
        mus)
    SCCAPML = StaticCCAProbabilisticModelLoader
    
    return [SCCAPML(W, Psi, mu, Z, lazy=lazy)
            for (W, Psi, mu) in zipped]

class Fu2016Loader:

    def __init__(self, Z, d, lazy=True, density=0.1):

        self.Z = Z
        self.d = d
        self.lazy = lazy
        self.density = density

        (self.k, self.num_data) = self.Z.shape
        self.data = None

        if not self.lazy:
            self._set_data()

        def get_data(self):

            if self.data is None:
                self._set_data()

            return self.data

        def _set_data(self):

            A = get_sparse_normal(d, k, density=density)

            self.data = np.dot(A, self.Z).T

class EventCCAProbabilisticModelLoader:

    def __init__(self, W, Psi, mu, Z, fakes, lazy=True):
        pass

class StaticCCAProbabilisticModelLoader:

    def __init__(self, W, Psi, mu, Z, lazy=True):
        
        self.W = W
        self.Psi = Psi
        self.mu = mu
        self.Z = Z
        self.lazy = lazy

        self.d = self.W.shape[0]
        (self.k, self.num_data) = self.Z.shape
        self.mean = np.dot(self.W, Z) + self.mu
        self.sd = get_svd_power(self.Psi, 0.5)
        self.data = None

        if not self.lazy:
            self._set_data()

    def get_data(self):

        if self.data is None:
            self._set_data()

        return self.data

    def _set_data(self):

        init = np.random.randn(
            self.num_data, self.d)
        shifted = init + self.mean.T

        self.data = np.dot(shifted, self.sd)
