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

    group2.add_argument('-alpha', '--alpha', help = 'Acceptance parameter of DPP', default = 0.9)
    group2.add_argument('-beta', '--beta', help = 'Acceptance parameter of DPP', default = 1.1)
    group2.add_argument('-gamma', '--gamma', help = 'Acceptance parameter of DPP', default = 0.1)
    group2.add_argument('-mu', '--mu', help = 'Acceptance parameter of DPP', default = 0.8)
    group2.add_argument('-eps', '--epsilon', help = 'Acceptance parameter of DPP', default = 0.1)

    args = vars(parser.parse_args())

    generator = ImageGenerator(args['images'], args['groundtruth'], args['detections'])
    
    cv2.namedWindow('DPP', cv2.WINDOW_NORMAL)
    #cv2.setWindowProperty('DPP', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    if args['images']:
        dpp = DPP(epsilon = float(args['epsilon']), mu = float(args['mu']), gamma = float(args['gamma']))
        for i in xrange(generator.get_sequences_len()):
            img = generator.get_frame(i)
            gt = generator.get_groundtruth(i)
            detections = generator.get_detections(i)
            weights = generator.get_detection_weights(i)
            features = generator.get_features(i)
            #print str(args['epsilon']) + ',' + str(args['mu']) + ',' + str(args['gamma'])
            '''
            detections = dpp.run(detections, weights, features)
            if detections:
                for det in detections:
                    cv2.rectangle(img, det.bbox.p_min, det.bbox.p_max, (0, 255, 0), 2)
            '''
            indices = dpp.run(detections, weights, features)
            
            for idx in indices:
                bbox = detections[idx].bbox
                x = bbox.p_min[0]
                y = bbox.p_min[1]
                width = bbox.p_max[0] - bbox.p_min[0]
                height = bbox.p_max[1] - bbox.p_min[1]
            
                print str(i + 1) + ',' + str('-1') + ',' + str(x) + ',' + str(y) + ',' \
                + str(width) + ',' + str(height) + ',' + str(weights[idx]) \
                + ',-1,-1,-1,' + ','.join(np.char.mod('%f', features[idx]))
            
            cv2.imshow('DPP', img)
            cv2.waitKey(1)
    
    #cv2.destroyWindow('MTT')