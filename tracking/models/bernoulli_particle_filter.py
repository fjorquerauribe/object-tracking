import numpy as np
import math
import cv2
from sklearn.preprocessing import normalize
from filterpy.monte_carlo import residual_resample
import utils
from frcnn import FasterRCNN

class BernoulliParticleFilter:
    DIM = 4 # x,y,width,height
    POS_STD_X = 3.0
    POS_STD_Y = 3.0
    SCALE_STD_WIDTH = 3.0
    SCALE_STD_HEIGHT = 3.0
    THRESHOLD = 100

    NEWBORN_PARTICLES = 0
    BIRTH_PROB = 0.0
    SURVIVAL_PROB = 1.0
    INITIAL_EXISTENCE_PROB = 0.99
    DETECTION_RATE = 0.9
    CLUTTER_RATE = 1.0
    LAMBDA_C = 20.0
    PDF_C = 1.6e-4

    states = []
    num_particles = 100
    reference = None
    detector = None
    initialized = False

    def __init__(self, num_particles):
        self.initialized = False
        self.num_particles = num_particles
        self.states = np.empty((self.num_particles, self.DIM), dtype = int) # try np.empty((shape), dtype=int)
        self.existence_prob = self.INITIAL_EXISTENCE_PROB
        self.initialized = False

    def is_initialized(self):
        return self.initialized

    def reinitialize(self):
        self.initialized = False

    def initialize(self, img, groundtruth):
        self.detector = FasterRCNN()
        self.reference = [groundtruth.x, groundtruth.y, groundtruth.width, groundtruth.height]
        (self.img_height, self.img_width, _) = img.shape # if BGR or RGB (height,width,n_channels) else if GRAY (height,width)

        # Samples
        self.states[:,0] = self.reference[0] + np.random.normal(0.0, self.POS_STD_X, self.num_particles)
        self.states[:,1] = self.reference[1] + np.random.normal(0.0, self.POS_STD_Y, self.num_particles)
        self.states[:,2] = self.reference[2] + np.random.normal(0.0, self.SCALE_STD_WIDTH, self.num_particles)
        self.states[:,3] = self.reference[3] + np.random.normal(0.0, self.SCALE_STD_HEIGHT, self.num_particles)

        self.states[:,0] = [ state[0] if (state[0] >= 0) else\
            0 for state in self.states]
        self.states[:,1] = [ state[1] if (state[1] >= 0) else\
            0 for state in self.states]
        self.states[:,2] = [ state[2] if (state[2] + state[0]) < self.img_width else\
            self.img_width - state[0] for state in self.states]
        self.states[:,3] = [ state[3] if (state[3] - state[1]) < self.img_height else\
            self.img_height - state[1] for state in self.states]

        # Set initial weights
        self.weights = np.ones((self.num_particles), dtype = float) * ( 1.0/float(self.num_particles))
        self.initialized = True
    
    def predict(self):
        if self.initialized:
            self.existence_prob = self.BIRTH_PROB * (1.0 - self.existence_prob) + (self.SURVIVAL_PROB * self.weights.sum() * self.existence_prob)
            # Predicted states
            self.states[:,0] = self.states[:,0] + np.random.normal(0.0, self.POS_STD_X, self.num_particles)
            self.states[:,1] = self.states[:,1] + np.random.normal(0.0, self.POS_STD_Y, self.num_particles)
            self.states[:,2] = self.states[:,2] + np.random.normal(0.0, self.SCALE_STD_WIDTH, self.num_particles)
            self.states[:,3] = self.states[:,3] + np.random.normal(0.0, self.SCALE_STD_HEIGHT, self.num_particles)

            self.states[:,0] = [ state[0] if (state[0] >= 0) else\
             0 for state in self.states]
            self.states[:,1] = [ state[1] if (state[1] >= 0) else\
             0 for state in self.states]
            self.states[:,2] = [ 0 if (state[2] < 0) else self.img_width - state[0] if\
             (state[2] + state[0]) > self.img_width else state[2] for state in self.states ]
            self.states[:,3] = [ 0 if (state[3] < 0) else self.img_height - state[1] if\
             (state[3] + state[1]) > self.img_height else state[3] for state in self.states ]
            
            '''for idx, state in enumerate(self.states):
                dx = np.random.normal(0.0, self.POS_STD_X)
                dy = np.random.normal(0.0, self.POS_STD_Y)
                dw = np.random.normal(0.0, self.SCALE_STD_WIDTH)
                dh = np.random.normal(0.0, self.SCALE_STD_HEIGHT)

                _x = min(max(state[0] + dw, 0), self.img_width)
                _y = min(max(state[1] + dh, 0), self.img_height)
                _w = min(max(state[2] + dw, 0), self.img_width)
                _h = min(max(state[3] + dh, 0), self.img_height)

                if ((_x + _w) < self.img_width) and ((_y + _h) < self.img_height) and (_w < self.img_width) and (_h < self.img_height)\
                    and (_x > 0) and (_y > 0) and (_w > 0) and (_h > 0):
                    self.states[idx, 0] = _x
                    self.states[idx, 1] = _y
                    self.states[idx, 2] = _w
                    self.states[idx, 3] = _h
                else:
                    self.states[idx, 2] = self.reference[2]
                    self.states[idx, 3] = self.reference[3]'''
            self.weights = self.SURVIVAL_PROB * self.existence_prob * self.weights
            
    
    def update(self, img, detections = None):
        if not detections:
            detections = self.detector.detect(img)
            for d in detections:
                cv2.rectangle(img, (d.bbox.x, d.bbox.y), (d.bbox.x + d.bbox.width, d.bbox.y + d.bbox.height), (0,0,255), 3)
        
        if len(detections) > 0:
            location_weights = np.empty(len(detections), dtype = float)
            for idx, det in enumerate(detections):
                obs = np.array([det.bbox.x, det.bbox.y, det.bbox.width, det.bbox.height])
                location_weights[idx] = math.exp(2.0 * (-1.0 + utils.intersection_over_union(self.reference, obs)))

            psi = np.empty((len(self.states),len(detections)), dtype = float)
            for i, state in enumerate(self.states):
                for j, det in enumerate(detections):
                    obs = np.array([det.bbox.x, det.bbox.y, det.bbox.width, det.bbox.height])
                    #print utils.intersection_over_union(state, obs)
                    psi[i,j] = location_weights[j] * math.exp(2.0 * (-1.0 + utils.intersection_over_union(state, obs)))
            
            tau = psi.sum(axis = 0)
            self.weights = self.weights * (1 - self.DETECTION_RATE) + psi.sum(axis = 1)/float(self.LAMBDA_C * self.PDF_C)
            self.resample()

    def resample(self):
        self.weights = normalize(self.weights[:,np.newaxis], axis = 0, norm = 'l1').ravel()
        self.states = self.states[residual_resample(self.weights)]
        self.weights = np.ones((self.num_particles), dtype = int) * ( 1/float(self.num_particles))

    def estimate(self, img, color = (255,255, 0), draw = False):
        estimate = np.mean(self.states, axis = 0, dtype = int)
        self.reference = estimate
        if draw:
            cv2.rectangle(img, (estimate[0], estimate[1]), (estimate[0] + estimate[2], estimate[1] + estimate[3]), color, 3)
        return estimate

    def draw_particles(self, img, color = (255, 0, 0)):
        for state in self.states:
            cv2.rectangle(img, (state[0], state[1]), (state[0] + state[2], state[1] + state[3]), color, 1)
