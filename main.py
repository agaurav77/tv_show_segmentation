from moviepy.editor import *
import numpy as np
from cue import Cue, Util

clip = VideoFileClip('1.mp4')
frame = clip.get_frame(Util.duration(0, 4, 2))
cues = Cue()

# preds, probs = cues.predict_yolo(frame)
# print(preds, probs)

# frame = clip.get_frame(Util.duration(0, 0, 31.5))
# prob = cues.detect_black_frame(frame)
# print(prob)

# clip.save_frame('test.jpg', Util.duration(4, 39, 11))
cues.generate_logos('cnn', clip, Util.duration(0, 3, 8),
  Util.duration(0, 3, 16), 0.5, 55, 134, 340, 376)

# for sec in np.arange(0, 60, 0.5):
#   frame = clip.get_frame(Util.duration(0, 0, sec))
#   matches = cues.match_template('images/cnn_logo.jpg', frame)
#   print("%.1f: %s" % (sec, ",".join(map(lambda x: "%.2f" % (x), matches))))