import numpy as np

from .gaussian import GaussianLoader

class FakePeriodicGaussianLoader:

    def __init__(self, 
        n, p, hertz,
        lazy=True,
        batch_size=None, 
        k=None, 
        mean=None):

        self.hertz = hertz
        self.loader = GaussianLoader(
            n,
            p, 
            lazy=lazy, 
            batch_size=batch_size, 
            k=k, 
            mean=None)

    def get_data(self):

        return self.loader.get_data()

    def name(self):

        return 'FakePeriodicGaussianLoader'

    def get_status(self):

        info = self.loader.get_status()

        info['hertz'] = self.hertz

        return info

    def finished(self):

        return self.loader.finished()

    def refresh(self):

        self.loader.refresh()

    def cols(self):
        
        return self.loader.cols()

    def rows(self):
        
        return self.loader.rows()

class EmbeddedCosineLoader:

    def __init__(self,
        n,
        p,
        phase=0,
        amplitude=1.0,
        period=2*np.pi,
        vertical_shift=0,
        index=0,
        period_noise=False,
        phase_noise=False,
        amplitude_noise=False):

        self.n = n
        self.p = p
        self.phase = phase
        self.amplitude = amplitude
        self.period = period
        self.vertical_shift = vertical_shift
        self.index = index
        self.period_noise = period_noise
        self.phase_noise = phase_noise
        self.amplitude_noise = amplitude_noise

        self.num_rounds = 0
        self.data = None

    def get_data(self):

        if self.data is None:
            self.data = np.random.randn(self.n, self.p)

            x = np.arange(self.n)
            with_period = x * 2 * np.pi / self.period
            with_phase = with_period + self.phase 
            wave = np.cos(with_phase)
            with_amp = self.amplitude * wave 
            with_vshift = with_amp + self.vertical_shift
            
            self.data[:,self.index] += with_vshift

        return self.data

    def name(self):

        return 'EmbeddedCosineLoader'

    def cols(self):

        return self.p

    def rows(self):

        return self.num_rounds

    def get_status(self):

        return {
            'p': self.p,
            'max_rounds': self.max_rounds,
            'phase': self.phase,
            'amplitude': self.amplitude,
            'period': self.period,
            'index': self.index,
            'num_rounds': self.num_rounds}
