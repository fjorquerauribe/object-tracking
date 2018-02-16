import numpy as np
import cv2
import argparse as ap

from utils.image_generator import MTTImageGenerator as ImageGenerator
from models.phd_filter import PHDFilter

if __name__ == '__main__':
    parser = ap.ArgumentParser()
    group1 = parser.add_mutually_exclusive_group(required = True)
    group1.add_argument('-i', '--images', help = 'Path to images folder')
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
        verbose = False
        idx = 1
        for i in xrange(generator.get_sequences_len()):
            img = generator.get_frame(i)
            gt = generator.get_groundtruth(i)
            detections = generator.get_detections(i)

            print detections
            for det in detections:
                cv2.rectangle(img, det.bbox, (0, 255, 0), 2)
            
            if verbose:
                print '-------------------------------------'
                print 'frame: ' + str(i)
                print 'groundtruth target: ' + str(len(gt))
            estimates = []
            if not filter.is_initialized():
                filter.initialize(img, detections)
                estimates = filter.estimate(img, draw = False, verbose = verbose)
                filter.draw_particles(img)
            else:
                filter.predict(verbose = verbose)
                filter.update(img, detections, verbose = verbose)
                estimates = filter.estimate(img, draw = False, verbose = verbose)
                filter.draw_particles(img)
            
            if estimates is not None:
                for e in estimates:
                    print str(idx) + ',' + str(e.label) + ',' + str(e.bbox.p_min[0]) + ',' + str(e.bbox.p_min[1]) + ','\
                    + str(e.bbox.p_max[0] - e.bbox.p_min[0]) + ',' + str(e.bbox.p_max[1] - e.bbox.p_min[1])\
                    + ',1,-1,-1,-1'
            idx+=1
            
            cv2.imshow('MTT', img)
            cv2.waitKey(1)
    
    #cv2.destroyWindow('MTT')