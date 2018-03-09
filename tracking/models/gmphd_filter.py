from utils import Target, Rectangle
from frcnn import FasterRCNN
import scipy.stats as stats
from scipy.optimize import linear_sum_assignment
import random
import numpy as np
import cv2

class GMPHDFilter:
    DIM = 4
    POS_STD_X = 3.0
    POS_STD_Y = 3.0
    SCALE_STD_WIDTH = 3.0
    SCALE_STD_HEIGHT = 3.0
    THRESHOLD = 10
    SURVIVAL_RATE = 1.0
    SURVIVAL_DECAY = 1.0
    CLUTTER_RATE = 2.0
    BIRTH_RATE = 0.1
    DETECTION_RATE = 0.5
    POSITION_LIKELIHOOD_STD = 30.0
    verbose = False
    initialized = False

    def __init__(self, verbose = False):
        self.verbose = verbose
        self.tracks = []
        self.labels = []
        self.detector = FasterRCNN()

    def is_initialized(self):
        return self.initialized

    def reinitialize(self):
        self.initialized = False

    def initialize(self, img):
        detections = self.detector.detect(img)
        (self.img_height, self.img_width, self.n_channels) = img.shape
        
        if len(detections) > 0:
            for det in detections:
                cv2.rectangle(img, det.bbox.p_min, det.bbox.p_max, (0, 255, 0), 2)
            
            for idx, det in enumerate(detections):
                # falta feature hist temporal
                x1 = int(det.bbox.p_min[0])
                y1 = int(det.bbox.p_min[1])
                x2 = int(det.bbox.p_max[0])
                y2 = int(det.bbox.p_max[1])
                crop_img = img[x1:x2, y1:y2]
                hist = cv2.calcHist( [ crop_img ], [0], None, [256], [0,256])
                #############################
                target = Target(det.bbox, idx, (random.randint(0,255), random.randint(0,255), random.randint(0,255)), det.conf, self.SURVIVAL_RATE, hist)
                self.tracks.append(target)
                self.labels.append(idx)
            self.initialized = True
    