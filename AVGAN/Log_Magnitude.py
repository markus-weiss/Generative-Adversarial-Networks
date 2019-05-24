import numpy as np
from math import *
import matplotlib.pyplot as plt

import scipy
from scipy.io import wavfile

fs, data = wavfile.read('test.wav')
# from amp_from_wave import main
import amp_from_wave
import os
#scipy.io.wavfile.read(somefile) returns a tuple of two items: the first is the
#sampling rate in samples per second, the second is a numpy array with all
# data read from the file.
from scipy.signal import stft #class to have the Fourrier's Transform from an image


#
#
#
#fs2 = 8e3
#N = 1e5
#amp = 2 * np.sqrt(2)
#noise_power = 0.01 * fs / 2
#time = np.arange(N) / float(fs)
#mod = 500*np.cos(2*np.pi*0.25*time)
#carrier = amp * np.sin(2*np.pi*3e3*time + mod)
#noise = np.random.normal(scale=np.sqrt(noise_power),size=time.shape)
#noise *= np.exp(-time/5)
#x = carrier + noise
#f, t, Zxx = stft(x, fs, nperseg=1000)
#
#logf=[0]*len(f)
#for i in range(1,(len(f))):
#    logf[i]=log10(f[i])
#plt.pcolormesh(t, logf, np.abs(Zxx), vmin=0, vmax=amp)
#plt.title('STFT log Magnitude')
#plt.ylabel('log10 Frequency [Hz]')
#plt.xlabel('Time [sec]')
#plt.show()

# amplitude=amp_from_wav('test.wav')

# amplitude=amp_from_wav('test.wav')

# amplitude =  os.system(amp_from_wav("test.wav"))

# amp_from_wave.getAmp('test.wav')

fs_rate, signal = wavfile.read("test.wav")
print ("Frequency sampling", fs_rate)
l_audio = len(signal.shape) #signal.shape=number of columns of 'signal'
print ("Channels", l_audio)
if l_audio == 2:
    signal = signal.sum(axis=1) / 2
N = signal.shape[0] #signal.shape[0] = number of lines of 'signal' (data)
print ("Complete Samplings N", N)
secs = N / float(fs_rate)
print ("secs", secs)
Ts = 1.0/fs_rate # sampling interval in time
print ("Timestep between samples Ts", Ts)
t = scipy.arange(0, secs, Ts) # time vector as scipy arange field / numpy.ndarray
FFT = abs(scipy.fft(signal))
FFT_side = FFT[range(N//2)] # one side FFT range
freqs = scipy.fftpack.fftfreq(signal.size, t[1]-t[0])
fft_freqs = np.array(freqs)
freqs_side = freqs[range(N//2)] # one side frequency range
fft_freqs_side = np.array(freqs_side)

print(secs)

plt.subplot(311)
p1 = plt.plot(t, signal, "g") # plotting the signal
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.subplot(312)
p2 = plt.plot(freqs, FFT, "r") # plotting the complete fft spectrum
plt.xlabel('Frequency (Hz)')
plt.ylabel('Count dbl-sided')
plt.subplot(313)
p3 = plt.plot(freqs_side, abs(FFT_side), "b") # plotting the positive fft spectrum
plt.xlabel('Frequency (Hz)')
plt.ylabel('Count single-sided')
plt.show()