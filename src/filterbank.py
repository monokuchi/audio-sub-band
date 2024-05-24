
import numpy as np
import numpy.typing as npt
from scipy import signal



class FilterBank:
    # Base Class for Filterbanks

    def __init__(self, N: int, P: int) -> None:
        self.N: int = N # Number of samples for each branch (also the FFT size), should be a power of 2
        self.P: int = P # Number of "subfilters" or "branches"



class PolyphaseFilterBank(FilterBank):
    # Danny C. Price, Spectrometers and Polyphase Filterbanks in Radio Astronomy, 2016. Available online at: http://arxiv.org/abs/1607.03579

    def __init__(self, N: int, P: int) -> None:
        super().__init__(N, P)

    def generate_window(self, window_type: str="hamming") -> npt.NDArray:
        # Generates a windowed sinc
        window_taps: npt.NDArray = signal.get_window(window_type, self.N*self.P)
        sinc: npt.NDArray = signal.firwin(self.N*self.P, 1.0/self.P, window="rectangular")
        window_taps *= sinc
        return window_taps



    def pfb_frontend(self, signal: npt.NDArray) -> npt.NDArray:
        num_windows: int = int(signal.shape[0] / (self.N * self.P))

        # Generate our window filter
        window: npt.NDArray = self.generate_window()

        # Apply our window filter onto our signal
        y: npt.NDArray = signal*window # Dimension of N x P

        # Split the result of signal*window into P branches via polyphase decomposition
        y = y.reshape(self.P, int(y.shape[0]/self.P))

        # Sum up our branches
        y = y.sum(axis=0)
        
        return y

    def pfb_filterbank(self, signal: npt.NDArray) -> npt.NDArray:
        frontend: npt.NDArray = self.pfb_frontend(signal)
        filterbank: npt.NDArray = np.fft.rfft(frontend, n=self.P)
        return filterbank




class MelFilterbank(FilterBank):
    def __init__(self, N: int, P: int) -> None:
        super().__init__(N, P)
