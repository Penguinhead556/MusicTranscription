import numpy as np
import librosa

y, sr = librosa.load("RED HEART.wav")

tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)

print(tempo)
