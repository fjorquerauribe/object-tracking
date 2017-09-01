import numpy as np
import cv2
import argparse as ap
from src.utils import utils
from src.models.particle_filter import ParticleFilter

if __name__ == '__main__':
    parser = ap.ArgumentParser()
    group1 = parser.add_mutually_exclusive_group(required=True)
    group1.add_argument('-i', '--images', help='Path to list of images')
    group1.add_argument('-v', '--video', help='Path to video file')
    group2 = parser.add_argument_group()
    group2.add_argument('-g', '--groundtruth', help='Path to groundtruth file', required=True)
    args = vars(parser.parse_args())

    cv2.namedWindow('Tracker', cv2.WINDOW_NORMAL)

    if args['images']:
        images = utils.read_images(args['images'])
        groundtruth = utils.read_groundtruth(args['groundtruth'])
        #print 'images len: ' + str(len(images))
        #print 'groundtruth len: ' + str(len(groundtruth))
        pf = ParticleFilter(100)
        for (img, g) in zip(images, groundtruth):
            cv2.rectangle(img, g.p_min, g.p_max, (255,0,0), 2)
            pf.initialize(img, g, True)
            cv2.imshow('Tracker', img)
            cv2.waitKey(1)
        cv2.destroyWindow('Tracker')
    elif args['video']:
        utils.read_video(args['video'])

    #pf.initialize(img,g)
