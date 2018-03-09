#!/usr/bin/python

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
    def __init__(self, image, region):
        initRegion = Rectangle(region.x, region.y, region.x + region.width, region.y + region.height)
        self.pf = BernoulliParticleFilter(100)
        self.pf.initialize(image, initRegion)

    def track(self, image):
        self.pf.predict()
        self.pf.update(image)
        estimate = self.pf.estimate(image, False)
        return vot.Rectangle(estimate[0], estimate[1], estimate[2], estimate[3])

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