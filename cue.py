import cv2
import numpy as np
import sys
from darknet import Darknet
from imageio import imsave
import os
import glob
import matplotlib.pyplot as plt
from pyAudioAnalysis import audioSegmentation

class Cue:

  def __init__(self):
    self.d = Darknet()

  # (internal)
  # return predictions using darknet.py that uses tiny yolo weights
  # (usage) preds, probs = cues._predict_yolo(frame)
  def _predict_yolo(self, frame):
    imsave('test.jpg', frame.astype('uint8'))
    preds = self.d.predictions('test.jpg')
    os.remove('test.jpg')
    return preds

  # (internal)
  # match a template image (say a cnn logo) in a video frame using
  # SIFT and FLANN; this function takes a single template image
  # and returns how many good descriptor matches happened
  def _match_template(self, template_img, frame):
    match_probs = []
    try:
      img1 = cv2.imread(template_img, 0)
      img2 = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
      sift = cv2.xfeatures2d.SIFT_create()
      kp1, des1 = sift.detectAndCompute(img1, None)
      kp2, des2 = sift.detectAndCompute(img2, None)
      index_params = dict(algorithm = 0, trees = 5)
      search_params = dict(checks = 50)
      flann = cv2.FlannBasedMatcher(index_params, search_params)
      matches = flann.knnMatch(des1, des2, k=2)
      for m,n in matches:
        ratio = m.distance/(n.distance*1.0)
        if ratio < 0.7:
          match_probs.append(ratio)
    except:
      pass
    return len(match_probs)

  # detect whether the given frame is mostly a black image or not, i.e.
  # a dark screen; threshold is the blackness percentage, above this
  # threshold, the frame is considered black and this func returns true
  # (usage) is_black_frame = cues.detect_black_frame(frame)
  def detect_black_frame(self, frame, threshold=0.95):
    retval, threshold = cv2.threshold(frame, 10, 255, cv2.THRESH_BINARY)
    size = np.size(threshold)*1.0
    num_black = np.sum(threshold == np.zeros(threshold.shape))
    return num_black/size >= threshold

  # generate mini templates for later use by looking at video clip "clip"
  # from "start" to "end" at "step" intervals, at the exact rectangle
  # from ["xs", "ys"] to ["xe", "ye"] and save all of these images at
  # images/"name"_*.jpg
  def generate_logos(self, name, clip, start, end, step, xs, xe, ys, ye):
    if os.path.isfile("images/%s_0.jpg" % (name)):
      print("images already exist, not regenerating")
      return
    idx = 0
    for timestep in np.arange(start, end+step, step):
      frame = clip.get_frame(timestep)
      frame = frame[ys:ye, xs:xe]
      imsave('images/%s_%d.jpg' % (name, idx), frame.astype('uint8'))
      idx += 1

  # find all the images in images directory that start with prefix
  def find_files(self, prefix):
    chosen_files = []
    idx = 0
    while True:
      image_name = 'images/%s_%d.jpg' % (prefix, idx)
      if os.path.isfile(image_name):
        chosen_files.append(image_name)
        idx += 1
      else:
        return chosen_files

  # match all images starting with the prefix to the given frame
  # returns true if the max number of hits was greater than or
  # equal to 6
  # TODO: can be made faster if we train yolo on these logos
  # (usage)
  # cues.generate_logos('cnn', clip, Util.duration(0, 3, 8),
  #   Util.duration(0, 3, 16), 0.5, 55, 134, 340, 376)
  # cnn_logos = cues.find_files('cnn')
  # logo_in_frame = cues.match_templates(cnn_logos, frame)
  def match_templates(self, files, frame):
    max_hit = 0
    for file in files:
      max_hit = max(max_hit, self._match_template(file, frame))
    return max_hit >= 6

  # (internal)
  # train and generate hmm model for speech music discrimination
  def _train_hmm(self):
    audioSegmentation.trainHMM_fromDir(
      "gtzan/", "weights/hmm", 1.0, 1.0)
    print('training complete, model at weights/hmm')

  # run speech or audio segmentation on audio clip from
  # using pyAudioAnalysis; assuming clip is already clipped
  def audio_classify(self, clip):
    if not os.path.isfile("weights/hmm"):
      print("pretrained model doesnt exist, training may take few minutes")
      self._train_hmm()
    else:
      print("pretrained model exists, using that")
    flags, classes, _, _ = audioSegmentation.hmmSegmentation(
      clip.to_soundarray(fps=44100), "weights/hmm")
    # classification is per second, lets group sections
    counts = []
    prev_item = None
    first_item = None
    for item in flags:
      if prev_item == None:
        counts.append(1)
        prev_item = item
        first_item = item
      elif item == prev_item:
        counts[-1] += 1
      elif item != prev_item:
        counts.append(1)
        prev_item = item
    counts = np.cumsum(np.array(counts))
    durations = [[], []]
    next_item = first_item
    for i in range(len(counts)):
      if i == 0:
        durations[next_item].append([0, counts[i]])
      else:
        durations[next_item].append([counts[i-1], counts[i]])
      next_item = (next_item+1)%2
    return dict(zip(classes, durations))

class Util:

  @staticmethod
  def duration(hours, mins, secs):
    return hours*60*60+mins*60+secs

  @staticmethod
  def inverse_duration(num):
    secs = num%60
    num //= 60
    mins = num%60
    num //= 60
    return num, mins, secs

  @staticmethod
  def show_predictions(res, c):
    # just show when music plays, rest is speech
    for item in res[c]:
      a1, a2, a3 = Util.inverse_duration(item[0])
      b1, b2, b3 = Util.inverse_duration(item[1])
      print("%d,%d,%d to %d,%d,%d" % (a1, a2, a3, b1, b2, b3))