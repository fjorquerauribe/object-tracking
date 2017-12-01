from utils.utils import Target, Rectangle, cost_matrix, appearance_affinity, motion_affinity, shape_affinity
import scipy.stats as stats
from scipy.optimize import linear_sum_assignment
import random
import numpy as np
import cv2

from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans

from detectors.resnet import Resnet

class PHDFilter:
    DIM = 4
    particles_batch = 0
    POS_STD_X = 3.0
    POS_STD_Y = 3.0
    SCALE_STD_WIDTH = 0.0
    SCALE_STD_HEIGHT = 0.0
    SURVIVAL_RATE = 0.99
    BIRTH_RATE = 5e-6
    POSITION_LIKELIHOOD = 30.0
    DETECTION_RATE = 0.5
    CLUTTER_RATE = 1.0e-3
    img_height = 0
    img_width = 0
    n_channels = 0
    persistent_states = []
    persistent_weights = []
    newborn_states = []
    newborn_weights = []
    tracks = []
    labels = set()
    current_labels = []
    birth_model = {}
    
    detector = None
    tracks_features = []
    
    initialized = False

    def __init__(self, particles_batch):
        self.particles_batch = particles_batch
        self.persistent_states = np.empty((0, self.DIM), dtype = int)
        self.newborn_states = np.empty((0, self.DIM), dtype = int)
        self.initialized = False

    def is_initialized(self):
        return self.initialized

    def reinitialize(self):
        self.initialized = False

    def initialize(self, img, detections):
        if len(detections) > 0:
            self.detector = Resnet()
            (self.img_height, self.img_width, self.n_channels) = img.shape
            dPosX = stats.norm( loc = 0.0, scale = self.POS_STD_X )
            dPosY = stats.norm( loc = 0.0, scale = self.POS_STD_Y )
            dWidth = stats.norm( loc = 0.0, scale = self.SCALE_STD_WIDTH )
            dHeight = stats.norm( loc = 0.0, scale = self.SCALE_STD_HEIGHT )

            self.persistent_states = np.empty((self.particles_batch * len(detections), self.DIM), dtype = int)
            idx = 0
            for det in detections:
                self.persistent_states[idx*self.particles_batch:(idx+1)*self.particles_batch,0] = det.bbox.p_min[0] + dPosX.rvs(self.particles_batch)
                self.persistent_states[idx*self.particles_batch:(idx+1)*self.particles_batch,1] = det.bbox.p_min[1] + dPosY.rvs(self.particles_batch)
                self.persistent_states[idx*self.particles_batch:(idx+1)*self.particles_batch,2] = det.bbox.p_max[0] - det.bbox.p_min[0] + dWidth.rvs(self.particles_batch)
                self.persistent_states[idx*self.particles_batch:(idx+1)*self.particles_batch,3] = det.bbox.p_max[1] - det.bbox.p_min[1] + dHeight.rvs(self.particles_batch)
                
                target = Target(det.bbox, idx, (random.randint(0,255), random.randint(0,255), random.randint(0,255)))
                self.tracks.append(target)
                self.current_labels.append(idx)
                self.labels.add(idx)
                idx+=1
            self.persistent_weights = np.ones((self.particles_batch * len(detections)), dtype = float) * ( 1.0/float(self.particles_batch))

            self.tracks_features = self.detector.get_features(img, self.tracks)
            #self.birth_model = detections
            self.initialized = True
    
    def predict(self, verbose = False):
        if self.initialized:
            self.persistent_states = np.append(self.persistent_states, self.newborn_states, axis = 0)
            self.persistent_weights = np.append(self.persistent_weights, self.newborn_weights)

            predicted_states = np.empty((0, self.DIM), dtype = int)
            predicted_weights = np.empty(0, dtype = float)
            for (state, weight) in zip(self.persistent_states, self.persistent_weights):
                if random.uniform(0, 1) <= self.SURVIVAL_RATE:
                    x = state[0] + int(round(stats.norm.rvs( loc = 0.0, scale = self.POS_STD_X )))
                    y = state[1] + int(round(stats.norm.rvs( loc = 0.0, scale = self.POS_STD_Y )))
                    w = state[2] + int(round(stats.norm.rvs( loc = 0.0, scale = self.SCALE_STD_WIDTH )))
                    h = state[3] + int(round(stats.norm.rvs( loc = 0.0, scale = self.SCALE_STD_HEIGHT )))
                    if x < 0:
                        x = 0
                    if y < 0:
                        y = 0
                    if(x + w) > self.img_width:
                        w = self.img_width - x
                    if(y + h) > self.img_height:
                        h = self.img_height - y
                        
                    predicted_states = np.append(predicted_states, [[x,y,w,h]], axis = 0)
                    predicted_weights = np.append(predicted_weights, (self.SURVIVAL_RATE * weight))
            self.persistent_states = predicted_states
            self.persistent_weights = predicted_weights

            if len(self.birth_model) > 0:
                dPosX = stats.norm( loc = 0.0, scale = self.POS_STD_X )
                dPosY = stats.norm( loc = 0.0, scale = self.POS_STD_Y )
                dWidth = stats.norm( loc = 0.0, scale = self.SCALE_STD_WIDTH )
                dHeight = stats.norm( loc = 0.0, scale = self.SCALE_STD_HEIGHT )
                self.newborn_states = np.empty((self.particles_batch * len(self.birth_model), self.DIM), dtype = int)
                self.newborn_weights = np.ones(self.particles_batch * len(self.birth_model), dtype = float) *\
                ( ( self.img_height * self.img_width * self.BIRTH_RATE ) / ( self.particles_batch * len(self.birth_model) ) )
                idx = 0
                for bm in self.birth_model:
                    self.newborn_states[idx*self.particles_batch:(idx+1)*self.particles_batch,0] = bm.bbox.p_min[0] + dPosX.rvs(self.particles_batch)
                    self.newborn_states[idx*self.particles_batch:(idx+1)*self.particles_batch,1] = bm.bbox.p_min[1] + dPosY.rvs(self.particles_batch)
                    self.newborn_states[idx*self.particles_batch:(idx+1)*self.particles_batch,2] = bm.bbox.p_max[0] - bm.bbox.p_min[0] + dWidth.rvs(self.particles_batch)
                    self.newborn_states[idx*self.particles_batch:(idx+1)*self.particles_batch,3] = bm.bbox.p_max[1] - bm.bbox.p_min[1] + dHeight.rvs(self.particles_batch) 
                    idx+=1
            if verbose:
                print 'predicted targets: ' + str(int(round(self.persistent_weights.sum())))

    def update(self, img, detections, verbose = False):
        if len(detections) > 0:
            self.birth_model = detections

            observations = np.empty((len(detections), self.DIM), dtype = float)
            idx = 0
            for det in detections:
                observations[idx, 0] = det.bbox.p_min[0]
                observations[idx, 1] = det.bbox.p_min[1]
                observations[idx, 2] = det.bbox.p_max[0] - det.bbox.p_min[0]
                observations[idx, 3] = det.bbox.p_max[1] - det.bbox.p_min[1]
                idx+=1
            
            psi = np.empty((len(self.persistent_states),len(detections)), dtype = float)
            cov = self.POSITION_LIKELIHOOD * self.POSITION_LIKELIHOOD * np.identity(4)
            idx = 0
            for (state, weight) in zip(self.persistent_states, self.persistent_weights):
                psi[idx,:] = stats.multivariate_normal.pdf(observations, state, cov) * self.DETECTION_RATE * weight
                idx+=1
            clutter_prob = self.CLUTTER_RATE / (self.img_height * self.img_width)
            tau = clutter_prob + np.sum(self.newborn_weights) + np.sum(psi, axis = 0)
            
            self.persistent_weights = (1.0 - self.DETECTION_RATE) * self.persistent_weights + (psi/tau).sum(axis = 1)
            
            if len(self.newborn_weights) > 0:
                tau = np.array([tau])
                self.newborn_weights = np.array([self.newborn_weights])
                self.newborn_weights = np.ones(len(self.newborn_states)) * (self.newborn_weights.T/tau).sum(axis = 1)

            if verbose:    
                print 'updated targets: ' + str(int(round(self.persistent_weights.sum())))
            self.resample()

    def resample(self):
        normalizedWeights = normalize(self.persistent_weights[:,np.newaxis], axis = 0, norm = 'l1').ravel()
        self.persistent_states = self.persistent_states[np.random.choice(np.arange(0,len(self.persistent_states)),\
         self.particles_batch * int(round(self.persistent_weights.sum())), replace = True, p = normalizedWeights)]
        self.persistent_weights = np.ones(len(self.persistent_states)) * (1.0/self.particles_batch)

        if len(self.newborn_weights) > 0:
            normalizedWeights = normalize(self.newborn_weights[:,np.newaxis], axis = 0, norm = 'l1').ravel()
            self.newborn_states = self.newborn_states[np.random.choice(np.arange(0, len(self.newborn_states)), \
             self.particles_batch * int(round(self.newborn_weights.sum())), replace = True, p = normalizedWeights)]
            N_b = self.particles_batch * len(self.birth_model)
            self.newborn_weights = np.ones(len(self.newborn_states)) * (self.newborn_weights.sum() / N_b)

    def estimate(self, img = None, draw = False, color = (0, 255, 0), verbose = False):
        if self.initialized:
            num_estimated = int(round(self.persistent_weights.sum()))
            
            if num_estimated > 0:
                kmeans = KMeans(n_clusters = num_estimated).fit(self.persistent_states)
                estimated_targets = np.rint(kmeans.cluster_centers_).astype(int)
                
                new_tracks = []
                label = 0
                for et in estimated_targets:
                    while (label in self.labels or label in self.current_labels):
                        label+=1
                    target = Target(Rectangle(et[0], et[1], et[0] + et[2], et[1] + et[3]), label,\
                    (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) )
                    new_tracks.append(target)
                    self.current_labels.append(label)
                    label+=1

                # affinity
                new_tracks_features = self.detector.get_features(img, new_tracks)
                affinity_matrix = appearance_affinity(self.tracks_features, new_tracks_features) *\
                                  motion_affinity(self.tracks, new_tracks) * \
                                  shape_affinity(self.tracks, new_tracks)
                affinity_matrix = 1./affinity_matrix
                row_ind, col_ind = linear_sum_assignment(affinity_matrix)
                #######
                #cost = cost_matrix(self.tracks, new_tracks)
                #row_ind, col_ind = linear_sum_assignment(cost)

                for r,c in zip(row_ind, col_ind):
                    new_tracks[c].color = self.tracks[r].color
                    new_tracks[c].label = self.tracks[r].label

                self.tracks = new_tracks[:]
                self.tracks_features = new_tracks_features[:]

                del self.current_labels[:]
                for track in self.tracks:
                    self.current_labels.append(track.label)
                self.labels = self.labels.union(self.current_labels)

                if draw:
                    for track in self.tracks:
                        cv2.rectangle(img, track.bbox.p_min, track.bbox.p_max, track.color, 2)
                if verbose:
                    print 'estimated targets: ' + str(num_estimated)
                return self.tracks

    def draw_particles(self, img, color = (255, 0, 0)):
        for state in self.persistent_states:
            cv2.rectangle(img, (state[0], state[1]), (state[0] + state[2], state[1] + state[3]), color, 2)