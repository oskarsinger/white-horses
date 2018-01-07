import os

import numpy as np

# TODO: cite dataset from UCI MLR
class AEGMMagnitudeLoader:

    def __init__(self, data_dir, subject, center=True, lazy=True):

        self.data_dir = data_dir
        self.subject = subject
        self.center = center
        self.lazy = lazy

        self.path = os.path.join(
            data_dir,
            'mHealth_subject' + str(self.subject) + '.log')
        self.data = None

        if not self.lazy:
            self._set_data() 

    def get_data(self):

        if self.data:
            self._set_data()

        return self.data

    def _set_data(self):

        raw_data = np.loadtxt(self.path)
        labels = raw_data[:,-1]
        mag_data = np.zeros((raw_data.shape[0], 9))

        mag_data[0,:] = np.linalg.norm(raw_data[:,:3], axis=1)
        mag_data[1,:] = raw_data[:,3]
        mag_data[2,:] = raw_data[:,4]
        mag_data[3,:] = np.linalg.norm(raw_data[:,5:8], axis=1)
        mag_data[4,:] = np.linalg.norm(raw_data[:,8:11], axis=1)
        mag_data[5,:] = np.linalg.norm(raw_data[:,11:14], axis=1)
        mag_data[6,:] = np.linalg.norm(raw_data[:,14:17], axis=1)
        mag_data[7,:] = np.linalg.norm(raw_data[:,17:20], axis=1)
        mag_data[8,:] = np.linalg.norm(raw_data[:,20:23], axis=1)

        mag_mean = np.mean(mag_data, axis=0)

        self.data = (mag_data - mag_mean, labels)

    def cols(self):

        return 9

    def rows(self):

        return self.data.shape[0]
