import numpy as np
import sounddevice as sd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import scipy

y, sr = librosa.load("RED HEART.wav")

downsample_factor = 4

# librosa.resample(y, sr, sr/downsample_factor)
# sr = sr/downsample_factor

# sd.play(y, sr)

tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, units='samples')

beat_frames = beat_frames - beat_frames[0]

print(beat_frames[0])

y = y[beat_frames[0]:]

winlength = int(sr*60/4/tempo)

D = librosa.stft(y, win_length = winlength, n_fft = 4096)

fig, ax = plt.subplots()
img = librosa.display.specshow(librosa.amplitude_to_db(np.abs(D),
                                                       ref=np.max),
                         y_axis='linear', x_axis='time', ax=ax)

maxIndicies  = []

print(len(D))

for i in range(len(D[0])-1):
    maxIndicies.append(scipy.signal.find_peaks(D[:,i], distance=200)[0])

tt = np.arange(0, int(len(y)/sr), 1/sr)

reconstructed = np.array([])

for i in range(len(beat_frames)-1):
    temp_arr = np.array([0]*winlength)
    for j in range(len(maxIndicies[i])):
        temp_arr = temp_arr + np.sin(tt[i*winlength:(i+1)*winlength]*maxIndicies[i][j] * 2 * np.pi)
    reconstructed = np.append(reconstructed, temp_arr)

# print(beat_frames[-1])
# print(len(reconstructed))

sd.play(reconstructed, sr)

plt.show()


