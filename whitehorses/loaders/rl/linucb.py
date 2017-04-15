import numpy as np

class LinUCBGaussianLoader:

    def __init__(self,
        inner_loader,
        ad,
        w=None,
        noise_variance=None,
        bias=False):

        self.loader = inner_loader
        self.ad = ad
        self.noise_variance = noise_variance
        self.noisy = noise_variance is not None
        self.bias = bias

        self.zd = self.loader.cols()
        self.ctd = self.ad * self.zd
        self.p = self.ad + self.zd + self.ctd
        
        if w is None:
            w = np.random.randn(self.p, 1)

        self.w = w
        self.a_history = []
        self.r_history = []
        self.num_rounds = 0

    def get_data(self):

        self.num_rounds += 1
        self.current_z = self.loader.get_data().T

        return np.copy(self.current_z)

    def set_action(self, a):

        self.a_history.append(a)

    def get_reward(self):

        x = np.vstack([
            self.current_z,
            self.a_history[-1]])
        r = np.dot(self.w.T, x)[0]

        if self.noisy:
            r += np.random.normal(
                scale=self.noise_variance)

        if np.any(r < 0):
            r = np.zeros_like(r)

        self.r_history.append(r)

        return r

    def get_max_reward(self, actions):

        xs = [np.vstack([self.current_z, a])
              for a in actions]

        return max(
            [np.dot(self.w.T, x)[0] for x in xs])

    def name(self):

        return 'LinUCBGaussianLoader'

    def cols(self):

        return self.zd

    def rows(self):

        return self.num_rounds
