import numpy as np

from linal.svd_funcs import get_svd_power

# TODO: make some util functions for generating special cases
# TODO: cite the Francis Bach paper

def get_Fu2016_Data():
    pass

def get_easy_CCAPMLs(num_data, k, ds, lazy=True):

    z = np.random.randn(k, num_data)
    Psi_inits = [np.random.randn(d * 2, d)
                 for d in ds]
    Psis = [np.dot(Pi.T, Pi) for Pi in Psi_inits]
    Ws = [np.random.randn(d, k) for d in ds]
    mus = [np.random.randn(d, 1) for d in ds]
    zipped = zip(
        Ws,
        Psis,
        mus)
    CCAPML = CCAProbabilisticModelLoader
    
    return [CCAPML(num_data, W, Psi, mu, z, lazy=lazy)
            for (W, Psi, mu) in zipped]

class CCAProbabilisticModelLoader:

    def __init__(self, num_data, W, Psi, mu, z, lazy=True):
        
        self.num_data = num_data
        self.W = W
        self.Psi = Psi
        self.mu = mu
        self.z = z
        self.lazy = lazy

        self.d = self.W.shape[0]
        self.k = self.z.shape[0]
        self.mean = np.dot(self.W, z) + self.mu
        self.sd = get_svd_power(self.Psi, 0.5)
        self.data = None

        if not self.lazy:
            self._set_data()

    def get_data(self):

        if self.data is None:
            self._set_data()

        return self.data

    def _generate_data(self):

        init = np.random.randn(
            self.num_data, self.d)
        shifted = init + self.mean.T

        self.data = np.dot(shifted, self.sd)
