import numpy as np

class PrivilegedInformationLoader:

    def __init__(self,
        regular_loader,
        privileged_loader,
        noisy=False,
        bias=False):

        self.rloader = regular_loader
        self.ploader = privileged_loader
        self.noisy = noisy
        self.bias = bias

        self.r_cols = self.rloader.cols()
        self.p_cols = self.ploader.cols()
        self.n = self.rloader.rows()

    def get_data(self):
        pass

    def name(self):

        return 'PrivilegedInformationLoader'

    def cols(self):

        return self.r_cols + self.p_cols

    def rows(self):

        return self.n
