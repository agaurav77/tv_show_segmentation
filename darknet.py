from ctypes import *
import math
import random

def sample(probs):
    s = sum(probs)
    probs = [a/s for a in probs]
    r = random.uniform(0, 1)
    for i in range(len(probs)):
        r = r - probs[i]
        if r <= 0:
            return i
    return len(probs)-1

def c_array(ctype, values):
    arr = (ctype*len(values))()
    arr[:] = values
    return arr

class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]

class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]

class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]

class Darknet:

    def __init__(self):
        self.lib = CDLL("darknet/libdarknet.so", RTLD_GLOBAL)
        self.lib.network_width.argtypes = [c_void_p]
        self.lib.network_width.restype = c_int
        self.lib.network_height.argtypes = [c_void_p]
        self.lib.network_height.restype = c_int
        
        self.predict = self.lib.network_predict
        self.predict.argtypes = [c_void_p, POINTER(c_float)]
        self.predict.restype = POINTER(c_float)
        
        self.set_gpu = self.lib.cuda_set_device
        self.set_gpu.argtypes = [c_int]

        self.make_image = self.lib.make_image
        self.make_image.argtypes = [c_int, c_int, c_int]
        self.make_image.restype = IMAGE

        self.make_boxes = self.lib.make_boxes
        self.make_boxes.argtypes = [c_void_p]
        self.make_boxes.restype = POINTER(BOX)

        self.free_ptrs = self.lib.free_ptrs
        self.free_ptrs.argtypes = [POINTER(c_void_p), c_int]

        self.num_boxes = self.lib.num_boxes
        self.num_boxes.argtypes = [c_void_p]
        self.num_boxes.restype = c_int

        self.make_probs = self.lib.make_probs
        self.make_probs.argtypes = [c_void_p]
        self.make_probs.restype = POINTER(POINTER(c_float))

        self.detect = self.lib.network_predict
        self.detect.argtypes = [c_void_p, IMAGE, c_float, c_float, c_float, POINTER(BOX), POINTER(POINTER(c_float))]

        self.reset_rnn = self.lib.reset_rnn
        self.reset_rnn.argtypes = [c_void_p]

        self.load_net = self.lib.load_network
        self.load_net.argtypes = [c_char_p, c_char_p, c_int]
        self.load_net.restype = c_void_p

        self.free_image = self.lib.free_image
        self.free_image.argtypes = [IMAGE]

        self.letterbox_image = self.lib.letterbox_image
        self.letterbox_image.argtypes = [IMAGE, c_int, c_int]
        self.letterbox_image.restype = IMAGE

        self.load_meta = self.lib.get_metadata
        self.lib.get_metadata.argtypes = [c_char_p]
        self.lib.get_metadata.restype = METADATA

        self.load_image = self.lib.load_image_color
        self.load_image.argtypes = [c_char_p, c_int, c_int]
        self.load_image.restype = IMAGE

        self.rgbgr_image = self.lib.rgbgr_image
        self.rgbgr_image.argtypes = [IMAGE]

        self.predict_image = self.lib.network_predict_image
        self.predict_image.argtypes = [c_void_p, IMAGE]
        self.predict_image.restype = POINTER(c_float)

        self.network_detect = self.lib.network_detect
        self.network_detect.argtypes = [c_void_p, IMAGE, c_float, c_float, c_float, POINTER(BOX), POINTER(POINTER(c_float))]

        # initialize
        self.net = self.load_net("darknet/cfg/tiny-yolo.cfg", "weights/tiny-yolo.weights", 0)
        self.meta = self.load_meta("darknet/cfg/coco.data")

    def classify(self, im):
        out = self.predict_image(self.net, im)
        res = []
        for i in range(self.meta.classes):
            res.append((self.meta.names[i], out[i]))
        res = sorted(res, key=lambda x: -x[1])
        return res

    def make_predictions(self, image, thresh=.5, hier_thresh=.5, nms=.45):
        im = self.load_image(image, 0, 0)
        boxes = self.make_boxes(self.net)
        probs = self.make_probs(self.net)
        num = self.num_boxes(self.net)
        self.network_detect(self.net, im, thresh, hier_thresh, nms, boxes, probs)
        res = []
        for j in range(num):
            for i in range(self.meta.classes):
                if probs[j][i] > 0:
                    res.append((self.meta.names[i], probs[j][i], (boxes[j].x, boxes[j].y, boxes[j].w, boxes[j].h)))
        res = sorted(res, key=lambda x: -x[1])
        self.free_image(im)
        self.free_ptrs(cast(probs, POINTER(c_void_p)), num)
        return res

    def predictions(self, image):
        return zip(*zip(*self.make_predictions(image))[0:2])