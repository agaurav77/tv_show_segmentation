from moviepy.editor import *
import numpy as np
from cue import Cue, Util

clip = VideoFileClip('1.mp4')
audio_clip = AudioFileClip('1.mp4')
cues = Cue()

start = Util.duration(2, 0, 0)
end = Util.duration(2, 10, 0)
clipped_audio = audio_clip.subclip(start, end)
results = cues.audio_classify(clipped_audio)
Util.show_predictions(results, 'music')