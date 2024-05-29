
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal
import filterbank



def plot_response(w, h, title):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(w, 20*np.log10(np.abs(h)))
    ax.set_ylim(-40, 5)
    ax.grid(True)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Gain (dB)')
    ax.set_title(title)



def padpower2(x: npt.NDArray):
    """Pads signal to the next highest power of 2

    Args:
        x (np.array): Input signal

    Returns:
        np.array: Padded signal
    """

    return np.pad(x, (0, int(2**np.ceil(np.log2(len(x)))) - len(x)), "constant")


def main():
    # Extract audio data
    sample_rate, audio_data = wavfile.read("../data/251_C.wav")
    audio_duration = audio_data.shape[0] / sample_rate
    print(f"Number of channels in WAV file: {audio_data.shape[1]}") # Should be 2 channels for stereo sound
    print(f"Audio Duration: {audio_duration}s")


    
    # Plot the audio stereo data in the time domain
    channel_time = np.linspace(0, audio_duration, audio_data.shape[0])
    plt.plot(channel_time, audio_data[:, 0], label="Left Channel")
    plt.plot(channel_time, audio_data[:, 1], label="Right Channel")
    plt.legend()
    plt.xlabel("Time [Sec]")
    plt.ylabel("Amplitude")
    plt.show()
    # plt.savefig("../figures/audio_plot_time.jpg")
    # plt.clf()




    P = 4
    N = 256
    padded_audio_data = padpower2(audio_data[:, 0])
    # padded_audio_data = np.sin(np.arange(0, N*P*10) / np.pi)
    polyphase_bank = filterbank.PolyphaseFilterBank(N, P)


    filter_banks = polyphase_bank.pfb_filterbank(padded_audio_data)

    concatenated_filter_banks = np.concatenate(filter_banks)
    polyphase_bank.graph_pfb(concatenated_filter_banks)


if __name__ == "__main__":
    main()
