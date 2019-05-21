import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
# import librosa

# (sig, rate) = librosa.load('testA.wav', sr=None)

# f= open("test.txt","r")
# print(f.read())

sample_rate, samples = wavfile.read('testA.wav')
frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)

plt.pcolormesh(times, frequencies, spectrogram)
plt.imshow(spectrogram)
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()