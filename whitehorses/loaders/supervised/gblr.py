import numpy as np

from theline.utils import get_multi_dot

# TODO: figure out how to make "low-rank" data for active learning
class BilinearLogisticRegressionLoader:

    def __init__(self,
        inner_loader1,
        inner_loader2,
        W=None,
        noisy=False):

        self.loader1 = inner_loader1
        self.loader2 = inner_loader2
        self.noisy = noisy
        self.bias = bias

        self.X1 = self.loader.get_data()
        self.X2 = self.loader.get_data()
        (self.n1, self.p1) = self.X1.shape
        (self.n2, self.p2) = self.X2.shape

        if W is None:
            W = np.random.randn(self.p1, self.p2)

        self.W = W
        inside_exp = get_multi_dot(
            [self.X1.T, self.W, self.X2]
        )
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

        return (self.X1, self.X2, self.y)


    def name(self):

        return 'BilinearLogisticRegressionLoader'


    def cols(self):

        return (self.p1, self.p2)


    def rows(self):

        return (self.n1, self.n2)
