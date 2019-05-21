# import wave, struct, math, random
# obj = wave.open('step.wav', 'r')

import tensorflow as tf

# audio_binary = tf.io.read_file('test.wav')
waveform = tf.audio.decode_wav(
    audio=,
    sample_rate=44100,
    name=None
)

# from IPython.display import Image, display
# ipd.Audio('./file_example_WAV_1MG')

# import librosa
# matplotlib inline
# import matplotlib.pyplot as plt
# import librosa.display

# data, sampling_rate = librosa.load('step.wav')
# print(sampling_rate)

# plt.figure(figsize=(12,4))
# librosa.display.waveplot(data, sr=sampling_rate)