import os
import cv2
import numpy as np

from utils import Rectangle, Detection

class BaseImageGenerator:
    def __init__(self):
        pass
    
    def read_images(self, path_to_images_list=''):
        if path_to_images_list=='':
            exit()
        with open(path_to_images_list, 'rb') as f:
            imageFiles = [line.strip() for line in f]

        path_to_images = os.path.dirname(path_to_images_list)
        images = []

        for imgFile in imageFiles:
            img = cv2.imread(path_to_images+'/'+imgFile) # BGR
            images.append(img)
        return images

    def read_video(self, video_file=''):
        pass

class Utils(BaseImageGenerator):
    def __init__(self):
        BaseImageGenerator.__init__(self)

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

class MTTImageGenerator():
    images = {}
    groundtruth = {}
    detections = {}
    def __init__(self, path_to_images='', path_to_groundtruth='', path_to_det=''):
        self.read_images(path_to_images)
        self.read_groundtruth(path_to_groundtruth)
        self.read_detections(path_to_det)
    
    def read_images(self, path_to_images = ''):
        if path_to_images == '':
            exit()
        idx = 1
        img = cv2.imread(path_to_images + str(idx).zfill(6) + '.jpg')
        while img is not None:
            self.images[idx-1] = img
            idx+=1
            img = cv2.imread(path_to_images + str(idx).zfill(6) + '.jpg')

    def read_detections(self, path_to_det = ''):
        if path_to_det == '':
            exit()
        
        with open(path_to_det, 'rb') as f:
            for line in f:
                line = line.split(',')
                line[-1] = line[-1].strip()
                num_frame = int(line[0])
                if num_frame - 1 not in self.detections:
                    self.detections[num_frame - 1] = []
                x_min = int(float(line[2]))
                y_min = int(float(line[3]))
                x_max = int(float(line[2]) + float(line[4]))
                y_max = int(float(line[3]) + float(line[5]))
                conf = float(line[6])
                detection = Detection(x_min, y_min, x_max, y_max, conf)
                self.detections[num_frame - 1].append(detection)

    def read_groundtruth(self, path_to_groundtruth = ''):
        if path_to_groundtruth == '':
            exit()
        
        with open(path_to_groundtruth, 'rb') as f:
            for line in f:
                line = line.split(',')
                line[-1] = line[-1].strip()
                num_frame = int(line[0])
                if num_frame - 1 not in self.groundtruth:
                    self.groundtruth[num_frame - 1] = []
                x_min = int(float(line[2]))
                y_min = int(float(line[3]))
                x_max = int(float(line[2]) + float(line[4]))
                y_max = int(float(line[3]) + float(line[5]))
                gt = Rectangle(x_min, y_min, x_max, y_max)
                self.groundtruth[num_frame - 1].append(gt)

    def get_sequences_len(self):
        return len(self.images)

    def get_frame(self, frame_num):
        return self.images[frame_num]

    def get_groundtruth(self, frame_num):
        if frame_num in self.groundtruth:
            return self.groundtruth[frame_num]
        else:
            return []

    def get_detections(self, frame_num):
        if frame_num in self.detections:
            return self.detections[frame_num]
        else:
            return []

    

