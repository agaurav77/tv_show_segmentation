import cv2
import numpy as np
import sys
from darknet import Darknet
from imageio import imsave
import os
import matplotlib.pyplot as plt

class Cue:

  def __init__(self):
    self.d = Darknet()

  def detect_black_frame(self, frame):
    retval, threshold = cv2.threshold(frame, 10, 255, cv2.THRESH_BINARY)
    size = np.size(threshold)*1.0
    num_black = np.sum(threshold == np.zeros(threshold.shape))
    return num_black/size

  def predict_yolo(self, frame):
    imsave('test.jpg', frame.astype('uint8'))
    preds = self.d.predictions('test.jpg')
    os.remove('test.jpg')
    return preds

  def match_template(self, template_img, frame):
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
    return sorted(match_probs)

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

class Util:

  @staticmethod
  def duration(hours, mins, secs):
    return hours*60*60+mins*60+secs