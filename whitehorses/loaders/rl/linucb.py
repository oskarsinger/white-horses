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
        self.actions = []
        self.rewards = []
        self.num_rounds = 0

    def get_data(self):

        self.current_x = self.loader.get_data().T
        self.num_rounds += 1

        return np.copy(self.current_x)

    def set_action(self, a):

        self.actions.append(a)

    def get_reward(self):

        x = np.vstack([
            self.current_x,
            self.actions[-1]])
        r = np.dot(self.w.T, x)[0,0]

        if self.noisy:
            r += np.random.randn()

        self.rewards.append(r)

        return r

    def name(self):

        return 'LinUCBGaussianLoader'

    def cols(self):

        return self.zd

    def rows(self):

        return self.num_rounds
