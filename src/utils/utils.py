import os
import cv2
import numpy as np
import math
from sklearn.metrics.pairwise import euclidean_distances

from scipy.optimize import linear_sum_assignment

class Rectangle:
     def __init__(self, x_min, y_min, x_max, y_max):
         self.p_min = (x_min,y_min)
         self.p_max = (x_max,y_max)

class Detection:
    conf = 0
    def __init__(self, x_min, y_min, x_max, y_max, conf):
        self.bbox = Rectangle(x_min, y_min, x_max, y_max)
        self.conf = conf

class Target:
    def __init__(self, bbox, label, color = None):
        self.bbox = bbox
        self.color = color
        self.label = label

def get_overlap_ratio(A, B):
    dx = min(A[0] + A[2], B[0] + B[2]) - max(A[0], B[0])
    dy = min(A[1] + A[3], B[1] + B[3]) - max(A[1], B[1])
    if (dx >= 0) and (dy >= 0):
        return (dx * dy) / float(B[2] * B[3])
    return 0.0

def bhatta(hist1, hist2):
    coef = np.sqrt(np.multiply(hist1,hist2)) / np.sqrt(np.sum(hist1) * np.sum(hist2))
    return np.sqrt(1 - np.sum(coef))

def cost_matrix(tracks, new_tracks):
    tracks_centroids = np.empty((0, 2), dtype = float)
    for t in tracks:
        tracks_centroids = np.append(tracks_centroids, [[ (t.bbox.p_max[0] + t.bbox.p_min[0])/2, (t.bbox.p_max[1] + t.bbox.p_min[1])/2 ]], axis = 0)
    
    new_tracks_centroids = np.empty((0, 2), dtype = float)
    for t in new_tracks:
        new_tracks_centroids = np.append(new_tracks_centroids, [[ (t.bbox.p_max[0] + t.bbox.p_min[0])/2, (t.bbox.p_max[1] + t.bbox.p_min[1])/2 ]], axis = 0)
    
    cost = euclidean_distances(tracks_centroids, new_tracks_centroids)
    return cost