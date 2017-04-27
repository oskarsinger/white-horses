import numpy as np

class LinUCBGaussianLoader:

    def __init__(self,
        inner_loader,
        ad,
        num_actions,
        ws=None,
        noise_variance=None,
        bias=False):

        self.loader = inner_loader
        self.ad = ad
        self.noise_variance = noise_variance
        self.noisy = noise_variance is not None
        self.num_actions = num_actions
        self.bias = bias

        self.zd = self.loader.cols()
        self.ctd = self.ad * self.zd
        self.p = self.ad + self.zd + self.ctd
        
        if ws is None:
            ws = [np.random.randn(self.p, 1)
                  for i in range(self.num_actions)]

        self.ws = ws
        self.a_history = []
        self.r_history = []
        self.num_rounds = 0

    def get_data(self):

        self.num_rounds += 1
        self.current_z = self.loader.get_data().T

        if self.noisy:
            self.current_noise = np.random.normal(
                scale=self.noise_variance)

        return np.copy(self.current_z)

    def set_action(self, a):

        self.a_history.append(a)

    def get_reward(self):

        (action, action_fs) = self.a_history[-1]
        x = np.vstack([
            self.current_z,
            action_fs])
        r = np.dot(self.ws[action].T, x)[0]

        if self.noisy:
            r += self.current_noise

        self.r_history.append(r)

        return r

    def get_max_reward(self, action_fs):

        xs = [np.vstack([self.current_z, af])
              for af in action_fs]
        rs = [np.dot(w.T, x)[0]
              for (x, w) in zip(xs, self.ws)]

        if self.noisy:
            rs = [r + self.current_noise
                  for r in rs]

        return max(rs)

    def name(self):

        return 'LinUCBGaussianLoader'

    def cols(self):

        return self.zd

    def rows(self):

        return self.num_rounds
