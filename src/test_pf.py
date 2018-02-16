import numpy as np
import cv2
import argparse as ap
from utils import utils
from models.particle_filter import ParticleFilter
from utils.image_generator import STTImageGenerator as ImageGenerator

import time

if __name__ == '__main__':
    parser = ap.ArgumentParser()
    group1 = parser.add_mutually_exclusive_group(required = True)
    group1.add_argument('-i', '--images', help='Path to images folder')
    group1.add_argument('-v', '--video', help='Path to video file')
    group2 = parser.add_argument_group()
    group2.add_argument('-g', '--groundtruth', help='Path to groundtruth file', required=True)
    group2.add_argument('-d', '--detections', help = 'Path to detections file', default = '')
    group2.add_argument('-npart', '--num_particles', help='Particle number of the Particle Filter')
    args = vars(parser.parse_args())

    generator = ImageGenerator(args['images'], args['groundtruth'], args['detections'])
    cv2.namedWindow('Tracker', cv2.WINDOW_NORMAL)
    
    if args['images']:
        pf = ParticleFilter(int(args['num_particles']))
        for i in xrange(generator.get_sequences_len()):
            img = generator.get_frame(i)
            gt = generator.get_groundtruth(i)
            
            grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            if not(pf.initialized):
                pf.initialize(grayImg, gt)
            else:
                pf.predict()
                pf.update(grayImg)
                pf.draw_particles(img)
                pf.estimate(img, True)
            
            cv2.rectangle(img, gt.p_min, gt.p_max, (0,0,255), 2)
            cv2.imshow('Tracker', img)
            cv2.waitKey(1)