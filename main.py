import numpy as np
from cue import Cue, Util

clip, audio_clip, ocr, subtitle = Util.get_clip_data('sample')
cues = Cue(ocr, subtitle)