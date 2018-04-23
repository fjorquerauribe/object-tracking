import os
import cv2
import numpy as np

from utils import Rectangle, Detection

class STTImageGenerator():
    images = {}
    groundtruth = {}
    detections = {}
    det_weights = {}
    features = {}
    FEATURES_DIM = 2048

    def __init__(self, path_to_images, path_to_groundtruth, path_to_det = None):
        self.read_images(path_to_images)
        self.read_groundtruth(path_to_groundtruth)
        if path_to_det:
            self.read_detections(path_to_det)

    def read_images(self, path_to_images = ''):
        if path_to_images == '':
            exit()
        idx = 1
        for file in sorted(os.listdir(path_to_images)):
            if file.endswith('.jpg'):
                self.images[idx - 1] = cv2.imread(path_to_images + file)
                idx+=1
    
    def read_groundtruth(self, path_to_groundtruth = ''):
        if path_to_groundtruth == '':
            exit()
        
        with open(path_to_groundtruth, 'rb') as f:
            num_frame = 1
            for line in f:
                line = line.rstrip().split(',')
                xs = map(float, line[::2])
                ys = map(float, line[1::2])
                x = int(min(xs))
                y = int(min(ys))
                width = int(max(xs)) - x
                height = int(max(ys)) - y
                gt = Rectangle( x, y, width, height )
                self.groundtruth[num_frame - 1] = gt
                num_frame+=1
    
    def read_detections(self, path_to_det = ''):
        if path_to_det == '':
            exit()
    
        with open(path_to_det, 'rb') as f:
            for line in f:
                line = line.split(',')
                line[-1] = line[-1].strip()
                
                num_frame = int(line[0])
                if num_frame not in self.detections:
                    self.detections[num_frame] = []
                    self.det_weights[num_frame] = np.empty(0)
                x = int(float(line[1]))
                y = int(float(line[2]))
                width = int(float(line[3]))
                height = int(float(line[4]))
                conf = float(line[5])
                self.det_weights[num_frame] = np.append(self.det_weights[num_frame], conf)
                detection = Detection(x, y, width, height, conf)
                self.detections[num_frame].append(detection)

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

class MTTImageGenerator():
    images = {}
    groundtruth = {}
    detections = {}
    det_weights = {}
    features = {}
    FEATURES_DIM = 0
    def __init__(self, path_to_images = None, path_to_groundtruth = None, path_to_det = None):
        if path_to_images is None:
            print 'No images path given ...'
            exit()
        else:
            self.read_images(path_to_images)
        if path_to_groundtruth is None:
            print 'No groundtruth path given ...'
            exit()
        else:
            self.read_groundtruth(path_to_groundtruth)
        if path_to_det is not None:
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
                num_frame = int(float(line[0]))
                if num_frame - 1 not in self.detections:
                    self.detections[num_frame - 1] = []
                    self.features[num_frame - 1] = np.empty((0, self.FEATURES_DIM))
                    self.det_weights[num_frame - 1] = np.empty(0)
                x = int(float(line[2]))
                y = int(float(line[3]))
                width = int(float(line[4]))
                height = int(float(line[5]))
                conf = float(line[6])
                self.det_weights[num_frame - 1] = np.append(self.det_weights[num_frame - 1], float(line[6]))
                feat = map(float, line[10:])
                detection = Detection(x, y, width, height, conf, np.array(feat))
                self.detections[num_frame - 1].append(detection)
                self.features[num_frame - 1] = np.append(self.features[num_frame - 1], np.array([feat]), axis = 0)

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
                #print int(float(line[1]))
                x = int(float(line[2]))
                y = int(float(line[3]))
                width = int(float(line[4]))
                height = int(float(line[5]))
                gt = Rectangle(x, y, width, height)
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

    def get_detection_weights(self, frame_num):
        if frame_num in self.det_weights:
            return self.det_weights[frame_num]
        else:
            return None

    def get_features(self, frame_num):
        if frame_num in self.features:
            return self.features[frame_num]
        else:
            return None