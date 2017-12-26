import numpy as np
import cv2
import argparse as ap

from utils.image_generator import MTTImageGenerator as ImageGenerator
from dpp.dpp import DPP

if __name__ == '__main__':
    parser = ap.ArgumentParser()
    group1 = parser.add_mutually_exclusive_group(required = True)
    group1.add_argument('-i', '--images', help = 'Path to images folder')
    group1.add_argument('-v', '--video', help = 'Path to video file')
    group2 = parser.add_argument_group()
    group2.add_argument('-g', '--groundtruth', help = 'Path to groundtruth file', required = True)
    group2.add_argument('-d', '--detections', help = 'Path to detections file', required = True)
    args = vars(parser.parse_args())

    generator = ImageGenerator(args['images'], args['groundtruth'], args['detections'])
    
    cv2.namedWindow('DPP', cv2.WINDOW_NORMAL)
    #cv2.setWindowProperty('DPP', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    if args['images']:
        verbose = False
        idx = 1
        dpp = DPP(epsilon = 0.05)
        for i in xrange(generator.get_sequences_len()):
            img = generator.get_frame(i)
            gt = generator.get_groundtruth(i)
            detections = generator.get_detections(i)
            weights = generator.get_detection_weights(i)
            features = generator.get_features(i)

            detections = dpp.run(detections, weights, features)

            if detections:
                for det in detections:
                    cv2.rectangle(img, det.bbox.p_min, det.bbox.p_max, (0, 255, 0), 2)

            cv2.imshow('DPP', img)
            cv2.waitKey(1)
    
    #cv2.destroyWindow('MTT')