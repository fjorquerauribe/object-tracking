import os
import cv2
import numpy as np

# class Rectangle:
#     def __init__(self, _p, _p2, _p3, _p4):
#         self.p1 = _p1
#         self.p2 = _p2
#         self.p3 = _p3
#         self.p4 = _p4

def read_images(path_to_images_list=''):
    if path_to_images_list=='':
        exit()
    with open(path_to_images_list, 'rb') as f:
        imageFiles = [line.strip() for line in f]

    path_to_images = os.path.dirname(path_to_images_list)
    images = []

    for imgFile in imageFiles:
        img = cv2.imread(path_to_images+'/'+imgFile)
        images.append(img)
    return images

def read_groundtruth(path_to_groundtruth=''):
    if path_to_groundtruth=='':
        exit()
    groundtruth = []
    with open(path_to_groundtruth, 'rb') as f:
        for line in f:
            line = line.split(',')
            line[-1] = line[-1].strip()
            points = np.array([ [int(float(line[0])),int(float(line[1]))], \
            [int(float(line[2])),int(float(line[3]))], [int(float(line[4])),int(float(line[5]))],\
            [int(float(line[6])),int(float(line[7]))] ], np.int32)
            points = points.reshape((-1,1,2))
            groundtruth.append(points)
    return groundtruth


def read_video(video_file=''):
    pass
