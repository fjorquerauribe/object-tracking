import os
import cv2
import numpy as np
import math
from sklearn.metrics.pairwise import euclidean_distances
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cosine

class Rectangle:
    def __init__(self, x_min, y_min, x_max, y_max):
        self.p_min = (x_min,y_min)
        self.p_max = (x_max,y_max)

class Detection:
    def __init__(self, x_min, y_min, x_max, y_max, conf = None, feature = None):
        self.bbox = Rectangle(x_min, y_min, x_max, y_max)
        self.conf = conf
        self.feature = feature

class Target:
    def __init__(self, bbox = None, label = None, color = None, conf = None, survival_rate = None, feature = None):
        self.bbox = bbox
        self.color = color
        self.label = label
        self.conf = conf
        self.survival_rate = survival_rate
        self.feature = feature

def get_overlap_ratio(A, B):
    dx = min(A[0] + A[2], B[0] + B[2]) - max(A[0], B[0])
    dy = min(A[1] + A[3], B[1] + B[3]) - max(A[1], B[1])
    if (dx >= 0) and (dy >= 0):
        return (dx * dy) / float(B[2] * B[3])
    return 0.0

def get_overlap_area(A, B):
    dx = min(A[0] + A[2], B[0] + B[2]) - max(A[0], B[0])
    dy = min(A[1] + A[3], B[1] + B[3]) - max(A[1], B[1])
    if (dx >= 0) and (dy >= 0):
        return dx * dy
    return 0.0

def intersection_over_union(A, B):
    xA = max( A[0], B[0] )
    yA = max( A[1], B[1] )
    xB = min( A[0] + A[2], B[0] + B[2] )
    yB = min( A[1] + A[3], B[1] + B[3] )
    if ((xB - xA + 1) < 0 or (yB - yA + 1) < 0):
        intersectionArea = 0
    else:    
        intersectionArea = (xB - xA + 1) * (yB - yA + 1)
    areaA = (A[2] + 1) * (A[3] + 1)
    areaB = (B[2] + 1) * (B[3] + 1)
    iou = max(intersectionArea / float(areaA + areaB - intersectionArea), 0)
    return iou

def bhatta(hist1, hist2):
    coef = np.sqrt(np.multiply(hist1,hist2)) / np.sqrt(np.sum(hist1) * np.sum(hist2))
    return np.sqrt(1 - np.sum(coef))

def cost_matrix(tracks, new_tracks, diagonal = 1.0, area = 1.0, norm = False):
    tracks_centroids = np.empty((0, 2), dtype = float)
    for t in tracks:
        tracks_centroids = np.append(tracks_centroids, [[ (t.bbox.p_max[0] + t.bbox.p_min[0])/2, (t.bbox.p_max[1] + t.bbox.p_min[1])/2 ]], axis = 0)
    
    new_tracks_centroids = np.empty((0, 2), dtype = float)
    for t in new_tracks:
        new_tracks_centroids = np.append(new_tracks_centroids, [[ (t.bbox.p_max[0] + t.bbox.p_min[0])/2, (t.bbox.p_max[1] + t.bbox.p_min[1])/2 ]], axis = 0)
    
    #print tracks_centroids
    #print new_tracks_centroids
    cost = euclidean_distances(tracks_centroids, new_tracks_centroids)
    
    if norm:
        #print cost
        #print '-------------------------------'
        
        feature_cost = np.zeros((len(tracks), len(new_tracks)), dtype = float)
        for idx1, t in enumerate(tracks):
            for idx2, nt in enumerate(new_tracks):
                #feature_cost[idx1, idx2] = ((t.feature - nt.feature)**2).sum()
                feature_cost[idx1, idx2] = cv2.compareHist(t.feature, nt.feature, cv2.HISTCMP_CORREL)

        position_cost = 1.0 + cost/diagonal
        scale_cost = 1.0 + cost/area
        
        #cost = feature_cost * position_cost * scale_cost
        cost = position_cost * scale_cost
        #print (feature_cost - feature_cost.min())/ (feature_cost.max() - feature_cost.min())
        #print '-----------------------------'
        #print cost
        #exit()
    
    return cost

def appearance_affinity(feat1, feat2):
    appearance_affinity_matrix = np.empty((feat1.shape[0], feat2.shape[0]))

    for i in xrange(feat1.shape[0]):
        for j in xrange(feat2.shape[0]):
            appearance_affinity_matrix[i,j] = cosine(feat1[i], feat2[j])
    return appearance_affinity_matrix

def motion_affinity(tracks1, tracks2, w = 0.1):
    motion_affinity_matrix = np.empty((len(tracks1), len(tracks2)))
    
    for i in xrange(len(tracks1)):
        for j in xrange(len(tracks2)):
            motion_affinity_matrix[i,j] = np.exp( -w * \
            ( np.power( (float((tracks1[i].bbox.p_min[0] - tracks2[j].bbox.p_min[0])) / (tracks2[j].bbox.p_max[0] - tracks2[j].bbox.p_min[0])), 2) \
            + np.power( (float((tracks1[i].bbox.p_min[1] - tracks2[j].bbox.p_min[1])) / (tracks2[j].bbox.p_max[1] - tracks2[j].bbox.p_min[1])), 2)))
    return motion_affinity_matrix

def shape_affinity(tracks1, tracks2, w = 0.1):
    shape_affinity_matrix = np.empty((len(tracks1), len(tracks2)))

    for i in xrange(len(tracks1)):
        for j in xrange(len(tracks2)):
            w_track1 = tracks1[i].bbox.p_max[0] - tracks1[i].bbox.p_min[0]
            h_track1 = tracks1[i].bbox.p_max[1] - tracks1[i].bbox.p_min[1]
            
            w_track2 = tracks2[j].bbox.p_max[0] - tracks2[j].bbox.p_min[0]
            h_track2 = tracks2[j].bbox.p_max[1] - tracks2[j].bbox.p_min[1]
            shape_affinity_matrix[i,j] = np.exp( -w * \
            ( ( float(np.absolute(h_track1 - h_track2))/(h_track1 + h_track2) )\
            + ( float(np.absolute(w_track1 - w_track2))/(w_track1 + w_track2) ) ))
    return shape_affinity_matrix

def nms(boxes, thresh, neighbors = 0, minScoresSum = 0.0):
    resRects = []
    idxs = []
    for idx, box in enumerate(boxes):
        idxs.append([box.conf, idx])
    idxs.sort()
    
    while len(idxs) > 0:
        lastElem = idxs.pop()
        rect1 = boxes[lastElem[1]].bbox
        x1 = rect1.p_min[0]
        y1 = rect1.p_min[1]
        w1 = rect1.p_max[0] - rect1.p_min[0]
        h1 = rect1.p_max[1] - rect1.p_min[1]
        
        neighborsCount = 0
        scoresSum = lastElem[0]

        for idx in idxs:
            rect2 = boxes[idx[1]].bbox
            x2 = rect2.p_min[0]
            y2 = rect2.p_min[1]
            w2 = rect2.p_max[0] - rect2.p_min[0]
            h2 = rect2.p_max[1] - rect2.p_min[1]

            iou = intersection_over_union([x1,y1,w1,h1], [x2,y2,w2,h2])
            if iou > thresh:
                scoresSum+= idx[0]
                idxs.remove(idx)
                neighborsCount+=1
        if neighborsCount >= neighbors and scoresSum >= minScoresSum:
            resRects.append(boxes[lastElem[1]])
    
    return resRects