import numpy as np

from linal.svd_funcs import get_svd_power

# TODO: cite the Francis Bach paper
class CCAProbabilisticModelLoader:

    def __init__(self, num_data, W, Phi, mu, z, lazy=True):
        
        self.num_data = num_data
        self.W = W
        self.Phi = Phi
        self.mu = mu
        self.z = z
        self.lazy = lazy

        self.d = self.W.shape[0]
        self.k = self.z.shape[0]
        self.mean = np.dot(self.W, z) + self.mu
        self.sd = get_svd_power(self.Phi, 0.5)
        self.data = None

        if not self.lazy:
            self.data = None

    def get_data(self):

        if self.data is None:
            init = np.random.randn(
                self.num_data, self.d)
            shifted = init + self.mean
            
            self.data = np.dot(shifted, self.sd)

        return self.data
