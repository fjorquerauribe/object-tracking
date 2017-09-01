from src.utils import utils
import scipy.stats as stats
import numpy as np
import cv2

# class State:
#     x = 0
#     y = 0
#     height = 0
#     width = 0
#     def __init__(self, x, y, width, height):
#         self.x = x
#         self.y = y
#         self.width = width
#         self.height = height
# class Rectangle:
#      def __init__(self, x_min, y_min, x_max, y_max):
#          self.p_min = (x_min,y_min)
#          self.p_max = (x_max,y_max)

class ParticleFilter:
    DIM = 4
    SIGMA = 20.0
    states = []
    weights = []
    num_particles = 0
    initialized = False

    def __init__(self, num_particles):
        self.initialized = False
        self.states = np.zeros((num_particles,self.DIM), dtype=int)
        self.num_particles = num_particles

    def is_initialized(self):
        return self.initialized

    def initialize(self, frame, groundtruth, draw=False):
        max_height = frame.shape[0]
        max_width = frame.shape[1]
        X = stats.truncnorm( 0.0, frame.shape[1], loc=groundtruth.p_min[0], scale=self.SIGMA )
        Y = stats.truncnorm( 0.0, frame.shape[0], loc=groundtruth.p_min[1], scale=self.SIGMA )
        self.states[:,0] = X.rvs(self.num_particles)
        self.states[:,1] = Y.rvs(self.num_particles)
        self.states[:,2] = np.full((self.num_particles),groundtruth.p_max[0])
        self.states[:,3] = np.full((self.num_particles),groundtruth.p_max[1])
        if draw == True:
            for state in self.states:
                print state
                cv2.rectangle(frame, (state[0],state[1]), (state[2], state[3]), (0,255,0), 2)
        self.initialized = True
