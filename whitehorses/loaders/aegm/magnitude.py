import numpy as np

# TODO: cite dataset from UCI MLR
class AEGMMagnitudeLoader:

    def __init__(self, path, subject, center=True, lazy=True):

        self.path = path
        self.subject = subject
        self.center = center
        self.lazy = lazy

        self.data = None

        if not self.lazy:
            self._set_data() 

    def get_data(self):

        if self.data:
            self._set_data()

        return self.data

    def _set_data(self):

        pass

    def cols(self):

        return 9

    def rows(self):

        return self.data.shape[0]
