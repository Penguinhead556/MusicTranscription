import numpy as np
import sounddevice as sd
import librosa
import librosa.display
import matplotlib.pyplot as plt

y, sr = librosa.load("RED HEART.wav")

downsample_factor = 4

# librosa.resample(y, sr, sr/downsample_factor)
# sr = sr/downsample_factor

# sd.play(y, sr)

tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, units='samples')

beat_frames = beat_frames - beat_frames[0]

print(beat_frames[0])

y = y[beat_frames[0]:]

winlength = int(tempo*sr/60/4)

D = librosa.stft(y, win_length = winlength, n_fft = 16384)

fig, ax = plt.subplots()
img = librosa.display.specshow(librosa.amplitude_to_db(np.abs(D),
                                                       ref=np.max),
                         y_axis='linear', x_axis='time', ax=ax)

maxIndicies  = []

print(len(D))

for i in range(len(D[0])-1):
    maxIndicies.append(np.argmin(D[:,i])*sr/winlength)
# plt.plot(maxIndicies)

tt = np.arange(0, int(len(y)/sr), 1/sr)

reconstructed = np.array([])

for i in range(len(beat_frames)-1):
    reconstructed = np.append(reconstructed, np.sin(tt[i*winlength:(i+1)*winlength]*maxIndicies[i]/sr * 2 * np.pi))

print(beat_frames[-1])
print(len(reconstructed))

sd.play(reconstructed, sr)

plt.show()


