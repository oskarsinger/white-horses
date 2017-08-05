import numpy as np

class LinearDynamicsSequenceLoader:

    def __init__(self,
        A,
        num_data,
        seed=None):

        self.A = A
        self.num_data = num_data
        
        if seed is None:
            seed = np.ones((self.A.shape[1], 1))

        self.seed = seed

        self._set_noiseless_Y()
        self._set_Y()

    def _set_noiseless_Y(self):

        y_list = [self.seed]

        for t in range(self.num_data):
            old_y = y_list[-1]
            new_y = np.dot(self.A, old_y)

            y_list.append(new_y)

        self.noiseless_Y = np.array(y_list[1:]).T

    def _set_Y(self):

        noise = np.random.randn(
            self.num_data, 
            self.seed.shape[0])

        self.Y = self.noiseless_Y + noise

    def get_data(self):

        return np.copy(self.Y)

    def rows(self):

        return self.Y.shape[0]

    def cols(self):

        return self.Y.shape[1]

    def refresh(self):

        self._set_Y()

class LinearDynamicalSystemLoader:

    def __init__(self,
        X_loader,
        A,
        B,
        C,
        D,
        seed=None):

        self.X_loader = X_loader
        self.A = A
        self.B = B
        self.C = C
        self.D = D

        if seed is None:
            seed = np.ones((self.A.shape[1], 1))

        self.seed = seed

        self.X = self.X_loader.get_data()
        self.BX = np.dot(B, self.X.T).T
        self.DX = np.dot(D, self.X.T).T

        self._set_H_and_noiseless_Y()

    def _set_H_and_noiseless_Y(self):

        h_list = [self.seed]
        y_list = []

        for t in range(self.X.shape[0]):
            old_h = h_list[-1]
            Ah = np.dot(self.A, old_h)
            Bx = self.BX[t,:]
            new_h = Ah + Dx
            Dx = self.DX[t,:]
            Ch = np.dot(self.C, new_h)
            new_y = Ch + Dx
            
            h_list.append(new_h)
            y_list.append(new_y)

        self.H = np.array(h_list[1:]).T
        self.noiseless_Y = np.array(y_list).T

    def _set_Y(self):

        (N, p) = self.noiseless_Y.shape
        noise = np.random.randn(N, p)

        self.Y = self.noiseless_Y + noise

    def get_data(self):

        return (np.copy(self.X), np.copy(self.Y))

    def cols(self):

        return self.X.shape[1] + self.Y.shape[1]

    def rows(self):

        return self.X_loader.rows()

    def refresh(self):

        self._set_Y()
