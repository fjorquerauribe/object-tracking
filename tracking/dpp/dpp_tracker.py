from utils.utils import Target, Rectangle
import random
import numpy as np
import cv2

class DPPMTTracker:
    alpha = 0.9
    beta = 1.1
    gamma = 0.1
    mu = 0.8
    epsilon = 0.1
    tracks = []
    initialized = False

    def __init__(self, alpha = 0.9, beta = 1.1, gamma = 0.1, mu = 0.8, epsilon = 0.1):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.mu = mu
        self.epsilon = epsilon

    def is_initialized(self):
        return self.initialized

    def reinitialize(self):
        self.initialized = False

    def initialize(self, detections):
        if len(detections) > 0:
            idx = 0
            for det in detections:
                target = Target(det.bbox, idx, (random.randint(0,255), random.randint(0,255), random.randint(0,255)))
                idx+=1
