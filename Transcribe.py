import numpy as np
import simpleaudio as sa


wave_obj = sa.WaveObject.from_wave_file("RED HEART.mp3")
play_obj = wave_obj.play()
play_obj.wait_done() 


