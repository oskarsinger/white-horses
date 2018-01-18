import os

import numpy as np

from whitehorses.loaders.aegm import get_view_name

# TODO: cite dataset from UCI MLR
class AEGMMagnitudeLoader:

    def __init__(self, 
        data_dir, 
        subject, 
        view,
        center=True, 
        lazy=True):

        self.data_dir = data_dir
        self.subject = subject
        self.view_idx = view
        self.view_name = get_view_name(self.view_idx)
        self.center = center
        self.lazy = lazy

        self.path = os.path.join(
            data_dir,
            'mHealth_subject' + str(self.subject) + '.log')
        self.hertz = 50
        self.data = None

        if not self.lazy:
            self._set_data() 

    def get_data(self):

        if self.data is None:
            self._set_data()

        return self.data

    def _set_data(self):

        raw_data = np.loadtxt(self.path)
        labels = raw_data[:,-1]
        mag_data = None

        if self.view_idx == 0:
            mag_data = np.linalg.norm(raw_data[:,:3], axis=1)
        elif self.view_idx == 1:
            mag_data = raw_data[:,3]
        elif self.view_idx == 2:
            mag_data = raw_data[:,4]
        elif self.view_idx == 3:
            mag_data = np.linalg.norm(raw_data[:,5:8], axis=1)
        elif self.view_idx == 4:
            mag_data = np.linalg.norm(raw_data[:,8:11], axis=1)
        elif self.view_idx == 5:
            mag_data = np.linalg.norm(raw_data[:,11:14], axis=1)
        elif self.view_idx == 6:
            mag_data = np.linalg.norm(raw_data[:,14:17], axis=1)
        elif self.view_idx == 7:
            mag_data = np.linalg.norm(raw_data[:,17:20], axis=1)
        elif self.view_idx == 8:
            mag_data = np.linalg.norm(raw_data[:,20:23], axis=1)
        elif self.view_idx == 9:
            mag_data = raw_data[:,23]

        if self.view_idx < 9:
            mag_mean = np.mean(mag_data)
            mag_data -= mag_mean

        self.data = mag_data[:,np.newaxis]

    def cols(self):

        return 1

    def rows(self):

        return self.data.shape[0]

    def name(self):

        return 'Subject' + str(self.subject) + \
            'View' + str(self.view_name) + \
            'AEGMMagnitudeLoader'
