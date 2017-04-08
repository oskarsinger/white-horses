import numpy as np

class LinUCBGaussianLoader:

    def __init__(self,
        inner_loader,
        ad,
        w=None,
        noisy=False,
        bias=False):

        self.loader = inner_loader
        self.ad = ad
        self.noisy = noisy 
        self.bias = bias

        self.zd = self.loader.cols()
        
        if w is None:
            w = np.random.randn(self.zd + self.ad, 1)

        self.w = w
        self.a_history = []
        self.r_history = []
        self.num_rounds = 0

    def get_data(self):

        self.num_rounds += 1
        self.current_x = self.loader.get_data().T

        return np.copy(self.current_x)

    def set_action(self, a):

        self.a_history.append(a)

    def get_reward(self):

        x = np.vstack([
            self.current_x,
            self.a_history[-1]])
        r = np.dot(self.w.T, x)[0]

        if self.noisy:
            r += np.random.randn()

        self.r_history.append(r)

        return r

    def name(self):

        return 'LinUCBGaussianLoader'

    def cols(self):

        return self.zd

    def rows(self):

        return self.num_rounds
