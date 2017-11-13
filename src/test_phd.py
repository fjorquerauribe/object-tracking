import numpy as np
import cv2
import argparse as ap

from utils.image_generator import MTTImageGenerator as ImageGenerator
from models.phd_filter import PHDFilter

if __name__ == '__main__':
    parser = ap.ArgumentParser()
    group1 = parser.add_mutually_exclusive_group(required = True)
    group1.add_argument('-i', '--images', help = 'Path to list of images')
    group1.add_argument('-v', '--video', help = 'Path to video file')
    group2 = parser.add_argument_group()
    group2.add_argument('-g', '--groundtruth', help = 'Path to groundtruth file', required = True)
    group2.add_argument('-d', '--detections', help = 'Path to detections file', required = True)
    group2.add_argument('-npart', '--particles_batch', help = 'Particle batch of the PHD Filter')
    args = vars(parser.parse_args())

    generator = ImageGenerator(args['images'], args['groundtruth'], args['detections'])
    cv2.namedWindow('MTT', cv2.WINDOW_NORMAL)
    
    if args['images']:
        filter = PHDFilter(int(args['particles_batch']))

        idx = 0
        for i in xrange(generator.get_sequences_len()):
            img = generator.get_frame(i)
            gt = generator.get_groundtruth(i)
            detections = generator.get_detections(i)
            
            print '-------------------------------------'
            print 'groundtruth target: ' + str(len(gt))
            if not filter.is_initialized():
                filter.initialize(img, detections)
                #filter.draw_particles(img)
            else:
                filter.predict()
                filter.update(img, detections)
                filter.estimate(img, draw = True)
                #filter.draw_particles(img)
            cv2.imwrite('./images/' + str(idx) + '.png', img)
            idx+=1
            
            cv2.imshow('MTT', img)
            cv2.waitKey(1)
    
    cv2.destroyWindow('MTT')