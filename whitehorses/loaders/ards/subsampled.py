import os

import numpy as np

from whitehorses.utils import get_one_hots as get_oh

class ARDSSubsampledEHRLUPILoader:

    def __init__(self, 
        csv_path,
        uncertain=False,
        lazy=True):

        self.csv_path = csv_path
        self.uncertain = uncertain
        self.lazy = lazy

        self.data = None

        if not self.lazy:
            self._set_data()

    def get_data(self):

        if self.data is None:
            self._set_data()

        return self.data

    def _set_data(self):

        with open(self.csv_path) as f:
            lines = [l.strip().split(',')
                     for l in f]
            as_numbers = [[float(i) for i in l]
                          for l in lines]
            as_numbers = [l if len(l) == 33 else l + [-1]
                          for l in as_numbers]
            as_np_array = np.array(as_numbers)
            pre_X_p = as_np_array[:,-1]

            pre_X_p += 1

            X_o = as_np_array[:,8:-1]
            X_p = np.zeros((X_o.shape[0], 8))

            to_one_hotify = pre_X_p[pre_X_p > 0].astype(int)
            X_p[pre_X_p > 0,:] = get_oh(to_one_hotify)

            y = as_np_array[:,2]
            c = as_np_array[:,3]

            if self.uncertain:
                self.data = (X_o, X_p, y, c)
            else:
                self.data = (X_o, X_p, y)

    def cols(self):

        cols = 0

        if self.data is not None:
            c_o = self.data[0].shape[1]
            c_p = self.data[1].shape[1]
            cols = c_o + c_p

        return cols

    def rows(self):

        rows = 0

        if self.data is not None:
            rows = self.data[0].shape[0]

        return rows

    def refresh(self):

        self.data = None
