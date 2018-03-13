import numpy as np
import cv2
import argparse as ap
from context import tracking
from tracking.utils import utils
from tracking.utils.image_generator import MTTImageGenerator as ImageGenerator
from tracking.detectors.frcnn import FasterRCNN

if __name__ == '__main__':
    parser = ap.ArgumentParser()
    group1 = parser.add_mutually_exclusive_group(required = True)
    group1.add_argument('-i', '--images', help = 'Path to images folder')
    group2 = parser.add_argument_group()
    group2.add_argument('-g', '--groundtruth', help = 'Path to groundtruth file', required = True)
    args = vars(parser.parse_args())
    generator = ImageGenerator(args['images'], args['groundtruth'])
    
    detector = FasterRCNN()
    cv2.namedWindow('Faster R-CNN', cv2.WINDOW_NORMAL)

    if args['images']:
        print generator.get_sequences_len()
        for i in xrange(generator.get_sequences_len()):
            img = generator.get_frame(i)
            gt = generator.get_groundtruth(i)
            detections = detector.detect(img)

            for d in detections:
                cv2.rectangle(img, d.bbox.p_min, d.bbox.p_max, (0,0,255), 2)
            
            cv2.imshow('Faster R-CNN', img)
            cv2.waitKey(1)