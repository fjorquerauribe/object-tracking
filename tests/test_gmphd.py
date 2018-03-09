import numpy as np
import cv2
import argparse as ap
from context import tracking
from tracking.utils.image_generator import MTTImageGenerator as ImageGenerator
from tracking.models.gmphd_filter import GMPHDFilter

if __name__ == '__main__':
    parser = ap.ArgumentParser()
    group = parser.add_argument_group()
    group.add_argument('-i', '--images', help = 'Path to images folder')
    group.add_argument('-g', '--groundtruth', help = 'Path to groundtruth file', required = True)
    args = vars(parser.parse_args())

    generator = ImageGenerator(args['images'], args['groundtruth'])
    
    cv2.namedWindow('MTT', cv2.WINDOW_NORMAL)
    
    if args['images']:
        filter = GMPHDFilter(False)
        verbose = False
        idx = 1
        for i in xrange(generator.get_sequences_len()):
            img = generator.get_frame(i)
            gt = generator.get_groundtruth(i)
            
            if verbose:
                print '-------------------------------------'
                print 'frame: ' + str(i)
                print 'groundtruth target: ' + str(len(gt))
            estimates = []
            if not filter.is_initialized():
                filter.initialize(img)
                #estimates = filter.estimate(img, draw = False, verbose = verbose)
                #filter.draw_particles(img)
            else:
                filter.initialize(img)
                #filter.predict(verbose = verbose)
                #filter.update(img, detections, verbose = verbose)
                #estimates = filter.estimate(img, draw = False, verbose = verbose)
                #filter.draw_particles(img)
            
            '''
            if estimates is not None:
                for e in estimates:
                    print str(idx) + ',' + str(e.label) + ',' + str(e.bbox.p_min[0]) + ',' + str(e.bbox.p_min[1]) + ','\
                    + str(e.bbox.p_max[0] - e.bbox.p_min[0]) + ',' + str(e.bbox.p_max[1] - e.bbox.p_min[1])\
                    + ',1,-1,-1,-1'
            idx+=1
            '''

            cv2.imshow('MTT', img)
            cv2.waitKey(1)
    
    #cv2.destroyWindow('MTT')