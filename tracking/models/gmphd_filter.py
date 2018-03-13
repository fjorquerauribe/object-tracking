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
        self.birth_model = []
        self.detector = FasterRCNN()

    def is_initialized(self):
        return self.initialized

    def reinitialize(self):
        self.initialized = False

    def initialize(self, img):
        detections = self.detector.detect(img)
        (self.img_height, self.img_width, self.n_channels) = img.shape
        self.tracks = []
        if len(detections) > 0:
            for det in detections:
                cv2.rectangle(img, det.bbox.p_min, det.bbox.p_max, (0, 255, 0), 2)
            
            for idx, det in enumerate(detections):
                # falta feature, hist es solo temporal
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

    def predict(self):
        if self.initialized:
            predicted_tracks = []
            for track in self.tracks:
                x = track.bbox.p_min[0] + int(round(stats.norm.rvs( loc = 0.0, scale = self.POS_STD_X )))
                y = track.bbox.p_min[1] + int(round(stats.norm.rvs( loc = 0.0, scale = self.POS_STD_Y )))
                w = track.bbox.p_max[0] - track.bbox.p_min[0] + int(round(stats.norm.rvs( loc = 0.0, scale = self.SCALE_STD_WIDTH )))
                h = track.bbox.p_max[1] - track.bbox.p_min[1] + int(round(stats.norm.rvs( loc = 0.0, scale = self.SCALE_STD_HEIGHT )))
                if x < 0:
                    x = 0
                if y < 0:
                    y = 0
                if(x + w) > self.img_width:
                    w = self.img_width - x
                if(y + h) > self.img_height:
                    h = self.img_height - y
                track.bbox.p_min = (x,y)
                track.bbox.p_max = (x + w, y + h)
                predicted_tracks.append(track)
            
            for track in self.birth_model:
                predicted_tracks.append(track)
            
            self.tracks = predicted_tracks

    def update(self, img, verbose = False):
        detections = self.detector.detect(img)
        if self.is_initialized() and len(detections) > 0:
            new_detections = []
            for det in detections:
                target = Target(bbox = det.bbox, color = (random.randint(0,255), random.randint(0,255), random.randint(0,255)),\
                 conf = det.conf, survival_rate = self.SURVIVAL_RATE)
                new_detections.append(target)
            self.tracks = new_detections


    def estimate(self, img = None, draw = False, color = (0, 255, 0)):
        if self.initialized:
            if draw:
                for track in self.tracks:
                    cv2.rectangle(img, track.bbox.p_min, track.bbox.p_max, track.color, 2)
            if self.verbose:
                print 'estimated targets: ' + str(len(self.tracks))
            return self.tracks