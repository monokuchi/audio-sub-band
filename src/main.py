
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



def padpower2(A):
    return np.pad(A, (0, int(2**np.ceil(np.log2(len(A)))) - len(A)), "constant")


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
    plt.savefig("../figures/audio_plot_time.jpg")
    plt.clf()





    # padded_audio_data = padpower2(audio_data[:, 0])

    P = 256
    N = 4
    padded_audio_data = np.sin(np.arange(0, N*P*2) / np.pi)
    polyphase_bank = filterbank.PolyphaseFilterBank(N, P)




    # plt.subplot(2, 1, 1)
    # plt.plot(padded_audio_data)
    # plt.subplot(2, 1, 2)
    # plt.plot(polyphase_bank.generate_window())
    # plt.show()

    p = polyphase_bank.pfb_filterbank(padded_audio_data)


    for pfp in p:
        polyphase_bank.graph_pfb(pfp)


if __name__ == "__main__":
    main()
