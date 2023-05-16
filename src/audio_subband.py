
import numpy as np
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


def main():
    # Extract audio data
    sample_rate, audio_data = wavfile.read("../data/251_C.wav")
    
    # # Plot the audio stereo data in the time domain
    # length = audio_data.shape[0] / sample_rate
    # time = np.linspace(0, length, audio_data.shape[0])
    # plt.plot(time, audio_data[:, 0], label="Left Channel")
    # plt.plot(time, audio_data[:, 1], label="Right Channel")
    # plt.legend()
    # plt.xlabel("Time [Sec]")
    # plt.ylabel("Amplitude")
    # plt.savefig("audio_plot_time.jpg")
    # plt.clf()

    # # Plot the audio stereo data in the frequency domain
    # plt.specgram(audio_data[:, 0], cmap="rainbow", Fs=sample_rate, label="Left Channel")
    # plt.specgram(audio_data[:, 1], cmap="ocean", Fs=sample_rate, label="Right Channel")
    # plt.legend()
    # plt.title("STFT Magnitude")
    # plt.xlabel("Time [Sec]")
    # plt.ylabel("Frequency [Hz]")
    # plt.savefig("audio_plot_freq.jpg")
    # plt.clf()

    # Create our analysis and synthesis banks
    analysis_bank = filterbank.Analysis(8, 64, sample_rate)
    synthesis_bank = filterbank.Synthesis(8, 64, sample_rate)
    
    # Plot one of the analysis filters
    w, h = signal.freqz(analysis_bank.retrieve(6), [1], fs=sample_rate)
    plot_response(w, h, "Analysis Filter")
    plt.savefig("analysis_filter.jpg")
    plt.clf()




if __name__ == "__main__":
    main()
