#!/usr/bin/python

#import time
from context import tracking
from tracking.models.bernoulli_particle_filter import BernoulliParticleFilter
from tracking.utils.utils import Rectangle
from tracking.vot import vot

import sys
import time
import cv2
import numpy
import collections

class BPFTracker:
    #pf = None
    def __init__(self, image, region):
        pass
        initRegion = Rectangle(region.x, region.y, region.x + region.width, region.y + region.height)
        self.pf = BernoulliParticleFilter(100)
        self.pf.initialize(image, initRegion)
        
        '''
        self.window = max(region.width, region.height) * 2

        left = max(region.x, 0)
        top = max(region.y, 0)

        right = min(region.x + region.width, image.shape[1] - 1)
        bottom = min(region.y + region.height, image.shape[0] - 1)

        self.template = image[int(top):int(bottom), int(left):int(right)]
        self.position = (region.x + region.width / 2, region.y + region.height / 2)
        self.size = (region.width, region.height)
        '''

    def track(self, image):
        self.pf.predict()
        self.pf.update(image)
        estimate = self.pf.estimate(image, False)
        return vot.Rectangle(estimate[0], estimate[1], estimate[2], estimate[3])
        
        '''
        left = max(round(self.position[0] - float(self.window) / 2), 0)
        top = max(round(self.position[1] - float(self.window) / 2), 0)

        right = min(round(self.position[0] + float(self.window) / 2), image.shape[1] - 1)
        bottom = min(round(self.position[1] + float(self.window) / 2), image.shape[0] - 1)

        if right - left < self.template.shape[1] or bottom - top < self.template.shape[0]:
            return vot.Rectangle(self.position[0] + self.size[0] / 2, self.position[1] + self.size[1] / 2, self.size[0], self.size[1])

        cut = image[int(top):int(bottom), int(left):int(right)]

        matches = cv2.matchTemplate(cut, self.template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(matches)

        self.position = (left + max_loc[0] + float(self.size[0]) / 2, top + max_loc[1] + float(self.size[1]) / 2)

        return vot.Rectangle(left + max_loc[0], top + max_loc[1], self.size[0], self.size[1])
        '''

handle = vot.VOT("rectangle")
selection = handle.region()

imagefile = handle.frame()
if not imagefile:
    sys.exit(0)

image = cv2.imread(imagefile)
tracker = BPFTracker(image, selection)
while True:
    imagefile = handle.frame()
    if not imagefile:
        break
    image = cv2.imread(imagefile)
    region = tracker.track(image)
    handle.report(region)

'''
# *****************************************
# VOT: Create VOT handle at the beginning
#      Then get the initializaton region
#      and the first image
# *****************************************
handle = vot.VOT("rectangle")
region = handle.region()

# Process the first frame
imagefile = handle.frame()
if not imagefile:
    sys.exit(0)

image = cv2.imread(imagefile)
tracker = BPFTracker(image, region)

while True:
    # *****************************************
    # VOT: Call frame method to get path of the 
    #      current image frame. If the result is
    #      null, the sequence is over.
    # *****************************************
    imagefile = handle.frame()
    if not imagefile:
        break

    image = cv2.imread(imagefile)
    region = tracker.track(image)
    
    # *****************************************
    # VOT: Report the position of the object 
    #      every frame using report method.
    # *****************************************
    
    handle.report(region)
'''