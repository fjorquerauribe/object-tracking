from utils import Target, Rectangle, cost_matrix, nms
#from resnet import Resnet
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
        #self.resnet = Resnet()

    def is_initialized(self):
        return self.initialized

    def reinitialize(self):
        self.initialized = False

    def initialize(self, img, detections = None):
        (self.img_height, self.img_width, self.n_channels) = img.shape
        self.tracks = []
        if len(detections) > 0 and detections:
            for det in detections:
                cv2.rectangle(img, det.bbox.p_min, det.bbox.p_max, (0, 255, 0), 2)
            #features = self.resnet.get_features(img, detections)
            for idx, det in enumerate(detections):
                # falta feature, hist es solo temporal
                x1 = int(det.bbox.p_min[0])
                y1 = int(det.bbox.p_min[1])
                x2 = int(det.bbox.p_max[0])
                y2 = int(det.bbox.p_max[1])
                crop_img = img[y1:y2, x1:x2]
                crop_img = cv2.resize(crop_img, (50,50))
                hist = cv2.calcHist( [ crop_img ], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256]).flatten()
                target = Target(det.bbox, idx, (random.randint(0,255), random.randint(0,255), random.randint(0,255)), det.conf, self.SURVIVAL_RATE, hist)
                #############################
                
                #target = Target(det.bbox, idx, (random.randint(0,255), random.randint(0,255), random.randint(0,255)), det.conf, self.SURVIVAL_RATE, features[idx,:])
                self.tracks.append(target)
                self.labels.append(idx)
            self.initialized = True

    def predict(self):
        if self.is_initialized():
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
                if np.random.uniform() < track.survival_rate:
                    track.bbox.p_min = (x,y)
                    track.bbox.p_max = (x + w, y + h)
                    predicted_tracks.append(track)
            
            for track in self.birth_model:
                predicted_tracks.append(track)
            
            self.tracks = predicted_tracks

    def update(self, img, detections = None, verbose = False):
        self.birth_model = []
        if self.is_initialized() and len(detections) > 0 and detections:
            #features = self.resnet.get_features(img, detections)
            new_detections = []
            for idx, det in enumerate(detections):
                # falta feature, hist es solo temporal
                x1 = int(det.bbox.p_min[0])
                y1 = int(det.bbox.p_min[1])
                x2 = int(det.bbox.p_max[0])
                y2 = int(det.bbox.p_max[1])
                crop_img = img[y1:y2, x1:x2]
                crop_img = cv2.resize(crop_img, (50,50))
                hist = cv2.calcHist( [ crop_img ], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256]).flatten()
                
                target = Target(bbox = det.bbox, color = (random.randint(0,255), random.randint(0,255), random.randint(0,255)),\
                 conf = det.conf, survival_rate = self.SURVIVAL_RATE, feature = hist)
                #############################
                
                #target = Target(bbox = det.bbox, color = (random.randint(0,255), random.randint(0,255), random.randint(0,255)),\
                # conf = det.conf, survival_rate = self.SURVIVAL_RATE, feature = features[idx,:])
                new_detections.append(target)
            
            diagonal = np.sqrt( np.power(self.img_height, 2) + np.power(self.img_width, 2) )
            area = self.img_height * self.img_width
            cost = cost_matrix(self.tracks, new_detections, diagonal, area, True)

            tracks_ind, new_dets_ind = linear_sum_assignment(cost)
            
            new_tracks = []
            for idxTrack, idxNewDet in zip(tracks_ind, new_dets_ind):
                if cost[idxTrack, idxNewDet] < self.THRESHOLD:
                    new_detections[idxNewDet].label = self.tracks[idxTrack].label
                    new_detections[idxNewDet].color = self.tracks[idxTrack].color
                    new_tracks.append(new_detections[idxNewDet])
                else:
                    self.tracks[idxTrack].survival_rate = np.exp(self.SURVIVAL_DECAY * (-1.0 + self.tracks[idxTrack].survival_rate * 0.9))
                    new_tracks.append(self.tracks[idxTrack])
            
            tracks_no_selected = set(np.arange(len(self.tracks))) - set(tracks_ind)
            for idxTrack in tracks_no_selected:
                self.tracks[idxTrack].survival_rate = np.exp(self.SURVIVAL_DECAY * (-1.0 + self.tracks[idxTrack].survival_rate * 0.9))
                new_tracks.append(self.tracks[idxTrack])
            
            new_detections_no_selected = set(np.arange(len(new_detections))) - set(new_dets_ind)
            for idxNewDet in new_detections_no_selected:
                if np.random.uniform() > self.BIRTH_RATE:
                    new_label = max(self.labels) + 1
                    new_detections[idxNewDet].label = new_label
                    self.birth_model.append(new_detections[idxNewDet])
                    self.labels.append(new_label)

            self.tracks = nms(new_tracks, 0.7, 0, 0.5)
            #self.tracks = new_tracks


    def estimate(self, img = None, draw = False, color = (0, 255, 0)):
        if self.initialized:
            if draw:
                for track in self.tracks:
                    cv2.rectangle(img, track.bbox.p_min, track.bbox.p_max, track.color, 3)
                    startX = track.bbox.p_min[0]
                    startY = track.bbox.p_min[1]
                    y = startY - 15 if startY - 15 > 15 else startY + 15
                    cv2.putText(img, str(track.label), (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, track.color, 2)
            if self.verbose:
                print 'estimated targets: ' + str(len(self.tracks))
            return self.tracks