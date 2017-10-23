import numpy as np

class PrivilegedInformationLoader:

    def __init__(self,
        observable_loader,
        privileged_loader,
        bias=False):

        self.oloader = obervable_loader
        self.ploader = privileged_loader
        self.bias = bias

        self.o_cols = self.oloader.cols()
        self.p_cols = self.ploader.cols()
        self.n = self.oloader.rows()
        self.wo = None
        self.wp = None

    def get_data(self):

        unnormed_X_o = self.oloader.get_data()
        X_p = self.ploader.get_data()
        Xw_p = np.dot(X_p, self.w_p)


        return (X_o, X_p, y)

    def name(self):

        return 'PrivilegedInformationLoader'

    def cols(self):

        return self.r_cols + self.p_cols

    def rows(self):

        return self.n
