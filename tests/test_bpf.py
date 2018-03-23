import numpy as np
import cv2
import argparse as ap

from context import tracking
from tracking.utils import utils
from tracking.models.bernoulli_particle_filter import BernoulliParticleFilter
from tracking.utils.image_generator import STTImageGenerator as ImageGenerator
from tracking.detectors.frcnn import FasterRCNN

if __name__ == '__main__':
    parser = ap.ArgumentParser()
    group1 = parser.add_mutually_exclusive_group(required = True)
    group1.add_argument('-i', '--images', help = 'Path to images folder')
    group1.add_argument('-v', '--video', help = 'Path to video file')
    group2 = parser.add_argument_group()
    group2.add_argument('-g', '--groundtruth', help = 'Path to groundtruth file', required = True)
    group2.add_argument('-d', '--detections', help = 'Path to detections file', default = '')
    group2.add_argument('-npart', '--num_particles', help = 'Particle number of the Bernoulli Particle Filter')
    args = vars(parser.parse_args())

    generator = ImageGenerator(args['images'], args['groundtruth'], args['detections'])

    cv2.namedWindow('Bernoulli Particle Filter', cv2.WINDOW_NORMAL)
    
    if args['images']:
        pf = BernoulliParticleFilter(int(args['num_particles']))
        for i in xrange(generator.get_sequences_len()):
            img = generator.get_frame(i)
            gt = generator.get_groundtruth(i)

            if not(pf.initialized):
                pf.initialize(img, gt)
            else:
                pf.predict()
                pf.update(img)
                pf.estimate(img, draw = True)

            cv2.rectangle(img, gt.p_min, gt.p_max, (0,255,0), 2)
            cv2.imshow('Bernoulli Particle Filter', img)
            cv2.waitKey(1)