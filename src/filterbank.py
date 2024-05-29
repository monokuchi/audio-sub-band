
import numpy as np
import numpy.typing as npt
from scipy import signal
import matplotlib.pyplot as plt



def db_power(signal: npt.NDArray) -> npt.NDArray:
    """Converts signal into dB scale

    Assumes that the signal is already in terms of power, i.e magnitude squared

    Args:
        signal (np.array): Input signal

    Returns:
        np.array: Signal in dB scale
    """

    return 10*np.log10(signal)


class FilterBank:
    # Base Class for Filterbanks

    def __init__(self, N: int, P: int) -> None:
        self.N: int = N # Number of polyphase sub-filters
        self.P: int = P # Number of taps in each polyphase sub-filters, also is the size of the FFT which we take at the end
        self.M: int = N * P # Size of each chunk of samples (we block process our signal). This requires our input signal to be some multiple of M
        self.L: int = 0 # Size of our input signal, must be a multiple of M
        self.W: int = int(self.L / self.M) # Number of data chunks we split our input signal into

    def __repr__(self) -> str:
        ret_str: str = f"N: {self.N}\n"
        ret_str += f"P: {self.P}\n"
        ret_str += f"M: {self.M}\n"
        ret_str += f"L: {self.L}\n"
        ret_str += f"W: {self.W}\n"
        return ret_str




class PolyphaseFilterBank(FilterBank):
    # Danny C. Price, Spectrometers and Polyphase Filterbanks in Radio Astronomy, 2016. Available online at: http://arxiv.org/abs/1607.03579

    def __init__(self, N: int, P: int) -> None:
        super().__init__(N, P)

    def generate_window(self, window_type: str="hamming") -> npt.NDArray:
        """Generates a windowed sinc of size M x 1 which will act as our "filter"

        Args:
            window_type (str): Type of window you want to generate (default "hamming")

        Returns:
            window (np.array): Window filter of size M x 1
        """

        window: npt.NDArray = signal.get_window(window_type, self.M)
        # sinc: npt.NDArray = signal.firwin(self.N*self.P, 1.0/self.P, window="rectangular")
        # We choose a sinc as our window function because we want the frequency response of each of our bins to be as close to a rect as possible
        sinc: npt.NDArray = np.sinc(np.linspace(-40, 40, num=self.M))
        window *= sinc
        return window.reshape(-1, 1) # Reshape into M x 1


    def pfb_frontend(self, signal: npt.NDArray) -> list[npt.NDArray]:
        """Polyphase filter frontend

        Args:
            signal (np.array): Input signal

        Returns:
            list[np.array]: List of each of our data chunks which are filtered, split up, then summed
        """

        self.L = signal.shape[0]
        self.W = int(self.L / self.M)

        print(self)
        # Generate our window filter -> M x 1
        W: npt.NDArray = self.generate_window()

        # Split our input signal into W data chunks each with size M via polyphase decomposition -> M x W
        X: npt.NDArray = signal.reshape(self.M, self.W)

        # Element wise multiply our window to each of our W data chunks
        Q: npt.NDArray = W * X # Windowed data -> M x W

        # Split a data chunk into N branches each with P samples and then sum up all the branches, do this for all W data chunks inside Q
        Y: list[npt.NDArray] = []
        for w in range(self.W):
            branches: list[npt.NPArray] = np.split(Q[:, w], self.N)
            summed_branches: npt.NPArray = sum(branches)
            Y.append(summed_branches)

        return Y


    def pfb_filterbank(self, signal: npt.NDArray) -> list[npt.NDArray]:
        """Returns a complete polyphase filterbank

        Args:
            signal (np.array): Input signal

        Returns:
            filterbank (list[np.array]): Our filterbank
        """

        frontend: list[npt.NDArray] = self.pfb_frontend(signal)
        filterbank: list[npt.NDArray] = [np.fft.rfft(i, n=self.P) for i in frontend]
        return filterbank


    def graph_pfb(self, pfb: npt.NDArray) -> None:
        """Graphs the power spectrum of a polyphase filterbank
        Args:
            pfb (np.array): Input polyphase filterbank

        Returns:
            None
        """

        # Get power spectral density of our fft
        pfb_psd: npt.NDArray = db_power(abs(pfb)**2)

        plt.plot(pfb_psd)
        plt.show()






class MelFilterbank(FilterBank):
    def __init__(self, N: int, P: int) -> None:
        super().__init__(N, P)
