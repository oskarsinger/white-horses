import numpy as np

from whitehorses.utils import get_minibatch

class LogisticRegressionGaussianLoader:

    def __init__(self,
        inner_loader,
        w=None,
        noisy=False,
        bias=False):

        self.loader = inner_loader
        self.noisy = noisy
        self.bias = bias

        self.X = self.loader.get_data()
        (self.n, self.p) = self.X.shape

        if self.bias:
            self.X = np.hstack(
                [self.X, np.ones((self.n, 1))])
            self.p += 1

        if w is None:
            w = np.random.randn(self.p, 1)

        self.w = w
        inside_exp = np.dot(self.X, self.w)
        self.probs = np.power(
            1 + np.exp(inside_exp),
            -1
        )

        if self.noisy:
            # TODO: figure this out later
            raise Exception(
                'Noisy bilinear logistic regression data not implemented.'
            )

        self.y = (self.probs > 0.5).astype(float)


    def get_data(self):

        return (self.X, self.y)


    def name(self):

        return 'LogisticRegressionGaussianLoader'


    def cols(self):

        return self.p

    def rows(self):

        return self.n


class ClassificationGaussianLoader:

    def __init__(self,
        inner_loader,
        w=None,
        noisy=False,
        bias=False):

        self.loader = inner_loader
        self.noisy = noisy
        self.bias = bias

        self.X = self.loader.get_data()
        (self.n, self.p) = self.X.shape

        if self.bias:
            self.X = np.hstack(
                [self.X, np.ones((self.n, 1))])
            self.p += 1

        if w is None:
            w = np.random.randn(self.p, 1)

        self.w = w
        self.y = np.dot(self.X, self.w)

        if self.noisy:
            self.noise = np.random.randn(self.n, 1)
            self.y += self.noise

        self.classes = np.sign(self.y)

    def get_data(self):

        return (self.X, self.classes)

    def name(self):

        return 'ClassificationGaussianLoader'

    def cols(self):

        return self.p

    def rows(self):

        return self.n

class LinearRegressionGaussianLoader:

    def __init__(self,
        inner_loader,
        w=None,
        noisy=False,
        bias=False):

        self.loader = inner_loader
        self.noisy = noisy
        self.bias = bias

        self.X = self.loader.get_data()
        (self.n, self.p) = self.X.shape

        if self.bias:
            self.X = np.hstack(
                [self.X, np.ones((self.n, 1))])
            self.p += 1

        if w is None:
            w = np.random.randn(self.p, 1)

        self.w = w
        self.y = np.dot(self.X, self.w)

        if self.noisy:
            self.noise = np.random.randn(self.n, 1)
            self.y += self.noise

    def get_data(self):

        return (self.X, self.y)

    def name(self):

        return 'LinearRegressionGaussianLoader'

    def cols(self):

        return self.p

    def rows(self):

        return self.n
