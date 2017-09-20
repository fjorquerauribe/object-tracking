from utils import utils
import scipy.stats as stats
import numpy as np
import cv2
from itertools import takewhile

# class Rectangle:
#      def __init__(self, x_min, y_min, x_max, y_max):
#          self.p_min = (x_min,y_min)
#          self.p_max = (x_max,y_max)

class ParticleFilter:
    DIM = 4 # x,y,width,height
    POS_STD_X = 1.0
    POS_STD_Y = 1.0
    SCALE_STD_WIDTH = 1.0
    SCALE_STD_HEIGHT = 1.0
    OVERLAP_RATIO = 0.90
    NEG_STD_X = 50.0
    NEG_STD_Y = 50.0
    states = []
    weights = []
    num_particles = 100
    reference = []
    initialized = False
    img_size = ()

    def __init__(self, num_particles):
        self.initialized = False
        self.num_particles = num_particles
        self.states = np.zeros((2 * self.num_particles,self.DIM), dtype=int)

    def is_initialized(self):
        return self.initialized

    def initialize(self, image, groundtruth, draw = False):
        self.reference = [groundtruth.p_min[0], groundtruth.p_min[1], \
            groundtruth.p_max[0] - groundtruth.p_min[0], \
            groundtruth.p_max[1] - groundtruth.p_min[1]]
        self.img_size = image.shape # (height,width,n_channels)

        # Positive samples
        dPosX = stats.norm( loc = 0.0, scale = self.POS_STD_X )
        DPosY = stats.norm( loc = 0.0, scale = self.POS_STD_Y )
        dWidth = stats.norm( loc = 0.0, scale = self.SCALE_STD_WIDTH )
        dHeight = stats.norm( loc = 0.0, scale = self.SCALE_STD_HEIGHT )

        self.states[:self.num_particles,0] = self.reference[0] + dPosX.rvs(self.num_particles)
        self.states[:self.num_particles,1] = self.reference[1] + DPosY.rvs(self.num_particles)
        self.states[:self.num_particles,2] = self.reference[2] + dWidth.rvs(self.num_particles)
        self.states[:self.num_particles,3] = self.reference[3] + dHeight.rvs(self.num_particles)

        self.states[:self.num_particles,2] = [ state[2] if (state[2] + state[0]) < self.img_size[1] else\
         max_width - state[0] for state in self.states[:self.num_particles,:]]
        self.states[:self.num_particles,3] = [ state[3] if (state[3] - state[1]) < self.img_size[0] else\
         max_height - state[1] for state in self.states[:self.num_particles,:]]

        # Negative samples
        dNegX = stats.norm( loc = 0.0, scale = self.NEG_STD_X)
        dNegY = stats.norm( loc = 0.0, scale = self.NEG_STD_Y)

        self.states[self.num_particles:,0] = self.reference[0] + dNegX.rvs(self.num_particles)
        self.states[self.num_particles:,1] = self.reference[1] + dNegY.rvs(self.num_particles)
        self.states[self.num_particles:,2] = self.reference[2]
        self.states[self.num_particles:,3] = self.reference[3]

        while True:
            ind = map(lambda state: utils.get_overlap_ratio(state, self.reference) > self.OVERLAP_RATIO, self.states[self.num_particles:,:])
            if ind:
                break
            self.states[self.num_particles:,][ind, 0] = self.reference[0] + dNegX.rvs(np.sum(ind))
            self.states[self.num_particles:,][ind, 1] = self.reference[1] + dNegX.rvs(np.sum(ind))

        if draw == True:
            for state in self.states[self.num_particles:,:]:
                cv2.rectangle(image, (state[0],state[1]), (state[0] + state[2], state[1] + state[3]), (255,0,0), 2)
            for state in self.states[:self.num_particles,:]:
                cv2.rectangle(image, (state[0],state[1]), (state[0] + state[2], state[1] + state[3]), (0,255,0), 2)
        self.initialized = True
