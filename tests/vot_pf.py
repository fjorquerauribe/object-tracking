#!/usr/bin/python

import time
from context import tracking
from tracking.vot import vot
from tracking.models.particle_filter import ParticleFilter
from tracking.utils.utils import Rectangle

class PFTracker:
    pf = None
    def __init__(self, image, region):
        initRegion = Rectangle(region.x, region.y, region.x + region.width, region.y + region.height)
        self.pf = ParticleFilter(100)
        self.pf.initialize(image, initRegion)

    def track(self, image):
        self.pf.predict()
        self.pf.update(image)
        estimate = self.pf.estimate(image, False)
        return vot.Rectangle(estimate[0], estimate[1], estimate[2], estimate[3])

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
tracker = PFTracker(image, region)

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