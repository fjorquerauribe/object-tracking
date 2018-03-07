import utils
import scipy.stats as stats
import numpy as np
import math
import cv2
from sklearn.preprocessing import normalize
from filterpy.monte_carlo import residual_resample

import warnings
import sys

THRESHOLD = 10

class ParticleFilter:
    DIM = 4 # x,y,width,height
    POS_STD_X = 1.0
    POS_STD_Y = 1.0
    SCALE_STD_WIDTH = 0.0
    SCALE_STD_HEIGHT = 0.0
    OVERLAP_RATIO = 0.2
    NEG_STD_X = 30.0
    NEG_STD_Y = 30.0
    states = []
    weights = []
    num_particles = 100
    reference = []
    histReference = []
    initialized = False
    img_width = 0
    img_height = 0
    num_channels = 0

    def __init__(self, num_particles):
        self.initialized = False
        self.num_particles = num_particles
        self.states = np.empty((self.num_particles,self.DIM), dtype=int) # try np.empty((shape), dtype=int)

    def is_initialized(self):
        return self.initialized

    def reinitialize(self):
        self.initialized = False

    def initialize(self, image, groundtruth):
        self.reference = [groundtruth.p_min[0], groundtruth.p_min[1], \
            groundtruth.p_max[0] - groundtruth.p_min[0], \
            groundtruth.p_max[1] - groundtruth.p_min[1]]
        (self.img_height, self.img_width, ) = image.shape # if BGR or RGB (height,width,n_channels) else if GRAY (height,width)

        # Positive samples
        dPosX = stats.norm( loc = 0.0, scale = self.POS_STD_X )
        dPosY = stats.norm( loc = 0.0, scale = self.POS_STD_Y )
        dWidth = stats.norm( loc = 0.0, scale = self.SCALE_STD_WIDTH )
        dHeight = stats.norm( loc = 0.0, scale = self.SCALE_STD_HEIGHT )

        self.states[:,0] = self.reference[0] + dPosX.rvs(self.num_particles)
        self.states[:,1] = self.reference[1] + dPosY.rvs(self.num_particles)
        self.states[:,2] = self.reference[2] + dWidth.rvs(self.num_particles)
        self.states[:,3] = self.reference[3] + dHeight.rvs(self.num_particles)

        self.states[:,0] = [ state[0] if (state[0] >= 0) else\
         0 for state in self.states]
        self.states[:,1] = [ state[1] if (state[1] >= 0) else\
         0 for state in self.states]
        self.states[:,2] = [ state[2] if (state[2] + state[0]) < self.img_width else\
         self.img_width - state[0] for state in self.states]
        self.states[:,3] = [ state[3] if (state[3] - state[1]) < self.img_height else\
         self.img_height - state[1] for state in self.states]

        '''
        self.states[:self.num_particles,:] = [ [self.reference[0] + _x, self.reference[1] + _y,\
            self.reference[2] + _width, self.reference[3] + _height] \
            if ( (self.reference[0] + _x + self.reference[2] + _width) and \
            (self.reference[1] + _y + self.reference[3] + _height) ) else self.reference \
            for (_x,_y,_width,_height) in zip(dPosX.rvs(self.num_particles), \
            dPosY.rvs(self.num_particles), dWidth.rvs(self.num_particles), \
            dHeight.rvs(self.num_particles))]
        '''
        '''
        # Negative samples
        dNegX = stats.norm( loc = 0.0, scale = self.NEG_STD_X)
        dNegY = stats.norm( loc = 0.0, scale = self.NEG_STD_Y)

        self.states[self.num_particles:,0] = self.reference[0] + dNegX.rvs(self.num_particles)
        self.states[self.num_particles:,1] = self.reference[1] + dNegY.rvs(self.num_particles)
        self.states[self.num_particles:,2] = self.reference[2]
        self.states[self.num_particles:,3] = self.reference[3]

        while True:
            ind = map(lambda state: utils.get_overlap_ratio(state, self.reference) > self.OVERLAP_RATIO, self.states[self.num_particles:,:])
            if np.sum(ind) == 0:
                break
            self.states[self.num_particles:,][ind, 0] = self.reference[0] + dNegX.rvs(np.sum(ind))
            self.states[self.num_particles:,][ind, 1] = self.reference[1] + dNegX.rvs(np.sum(ind))

        if draw == True:
            for state in self.states[self.num_particles:,:]:
                cv2.rectangle(image, (state[0],state[1]), (state[0] + state[2], state[1] + state[3]), (255,0,0), 1)
            for state in self.states[:self.num_particles,:]:
                cv2.rectangle(image, (state[0],state[1]), (state[0] + state[2], state[1] + state[3]), (0,255,0), 1)
        '''
        # Set initial weights
        self.weights = np.ones((self.num_particles), dtype = float) * ( 1/float(self.num_particles))

        self.histReference = cv2.calcHist( [ image[self.reference[0]:self.reference[0]+self.reference[2],\
        self.reference[1]:self.reference[1]+self.reference[3]]], [0], None, [256], [0,256])

        self.initialized = True

    def predict(self):
        dRandomX = stats.norm( loc = 0.0, scale = self.POS_STD_X )
        dRandomY = stats.norm( loc = 0.0, scale = self.POS_STD_Y )
        dRandomWidth = stats.norm( loc = 0.0, scale = self.SCALE_STD_WIDTH )
        dRandomHeight = stats.norm( loc = 0.0, scale = self.SCALE_STD_HEIGHT)

        if self.initialized:
            self.states[:,0] = self.states[:,0] + dRandomX.rvs(self.num_particles)
            self.states[:,1] = self.states[:,1] + dRandomY.rvs(self.num_particles)
            self.states[:,2] = self.states[:,2] + dRandomWidth.rvs(self.num_particles)
            self.states[:,3] = self.states[:,3] + dRandomHeight.rvs(self.num_particles)

            self.states[:,0] = [ state[0] if (state[0] >= 0) else\
             0 for state in self.states]
            self.states[:,1] = [ state[1] if (state[1] >= 0) else\
             0 for state in self.states]
            self.states[:,2] = [ state[2] if (state[2] + state[0]) < self.img_width else\
             self.img_width - state[0] for state in self.states]
            self.states[:,3] = [ state[3] if (state[3] - state[1]) < self.img_height else\
             self.img_height - state[1] for state in self.states]

    def update(self, image):
        for (state, idx) in zip(self.states, xrange(len(self.weights))):
            crop_img = image[state[0]:state[0]+state[2], state[1]:state[1]+state[3]]
            resized_img = cv2.resize(crop_img, (self.reference[2], self.reference[3]))
            hist = cv2.calcHist( [ crop_img ],\
             [0], None, [256], [0,256])
            self.weights[idx] = math.exp(-1.0 * cv2.compareHist(self.histReference, hist, cv2.HISTCMP_BHATTACHARYYA)) #cv2.HISTCMP_CHISQR
            print cv2.compareHist(self.histReference, hist, cv2.HISTCMP_BHATTACHARYYA)#self.weights[idx]
        self.resample()

    def resample(self):
        self.weights = normalize(self.weights[:,np.newaxis], axis = 0, norm='l1').ravel()
        #print np.square(self.weights)
        ESS = 1/float(np.sum(np.square(self.weights)))
        #print ESS,THRESHOLD
        if ESS > THRESHOLD:
            newStates = np.empty((len(self.weights), self.DIM), dtype=int)
            self.states = self.states[residual_resample(self.weights)]
            self.weights = np.ones((self.num_particles), dtype = int) * ( 1/float(self.num_particles))

    def estimate(self, image, color = (255,255, 0), draw = True):
        estimate = np.mean(self.states, axis = 0, dtype=int)
        if draw:
            cv2.rectangle(image, (estimate[0], estimate[1]), (estimate[0] + estimate[2], estimate[1] + estimate[3]), color, 1)
        return estimate

    def draw_particles(self, image, color = (255,0,0)):
        for state in self.states:
            cv2.rectangle(image, (state[0], state[1]), (state[0] + state[2], state[1] + state[3]), color, 1)
