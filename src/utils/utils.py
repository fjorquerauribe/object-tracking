import os
import cv2
import numpy as np

class Rectangle:
     def __init__(self, x_min, y_min, x_max, y_max):
         self.p_min = (x_min,y_min)
         self.p_max = (x_max,y_max)

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
            x_min = points[0][0]
            y_min = points[0][1]
            x_max = points[0][0]
            y_max = points[0][1]
            for p in points:
                if p[0] < x_min:
                    x_min = p[0]
                if p[1] < y_min:
                    y_min = p[1]
                if p[0] > x_max:
                    x_max = p[0]
                if p[1] > y_max:
                    y_max = p[1]
            rect = Rectangle(x_min, y_min, x_max, y_max)
            groundtruth.append(rect)
    return groundtruth

def get_overlap_ratio(A, B):
    dx = min(A[0] + A[2], B[0] + B[2]) - max(A[0], B[0])
    dy = min(A[1] + A[3], B[1] + B[3]) - max(A[1], B[1])
    if (dx >= 0) and (dy >= 0):
        return (dx * dy) / float(B[2] * B[3])
    return 0.0

def read_video(video_file=''):
    pass
