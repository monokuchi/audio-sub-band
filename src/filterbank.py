
import numpy as np
from scipy import signal


class FilterBank:
    def __init__(self, N, num_taps):
        self.N = N # Number of sub-bands
        self.num_taps = num_taps # Number of filter coefficents
        self.filters = np.zeros((N, num_taps)) # Array of bandpass filters

    def retrieve(self, k):
        return self.filters[k]


class Analysis(FilterBank):
    def __init__(self, N, num_taps, f_s):
        super().__init__(N, num_taps)

        # Make the bandpass filters
        for n in range(N):
            self.filters[n] = signal.remez(num_taps, [0, (((n/N)*20000)+20)-.1, ((n/N)*20000)+20, ((n+1)/N)*20000, (((n+1)/N)*20000)+.1, 20010], [0, 1, 0], fs=f_s, type="bandpass")
            

class Synthesis(FilterBank):
    def __init__(self, N, num_taps, f_s):
        super().__init__(N, num_taps)

        # Make the bandpass filters
        for n in range(N):
            self.filters[n] = signal.remez(num_taps, [0, (((n/N)*20000)+20)-.1, ((n/N)*20000)+20, ((n+1)/N)*20000, (((n+1)/N)*20000)+.1, 20010], [0, 1, 0], fs=f_s, type="bandpass")
    