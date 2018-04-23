import numpy as np
import math
from utils import get_overlap_area, get_overlap_ratio, appearance_affinity, motion_affinity, shape_affinity

class DPP:

    def __init__(self, alpha = 0.9, beta = 1.1, gamma = 0.1, mu = 0.8, epsilon = 0.1):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.mu = mu
        self.epsilon = epsilon

    def run(self, boxes = None, weights = None, features = None, img_size = None):
        if (len(boxes) > 0) and (boxes is not None):
            if (weights is not None) and (features is not None):
                area = np.empty(len(boxes), dtype = float)
                intersection = np.empty((len(boxes), len(boxes)) , dtype = float)
                for i in xrange(len(boxes)):
                    x1 = boxes[i].bbox.x
                    y1 = boxes[i].bbox.y
                    w1 = boxes[i].bbox.width
                    h1 = boxes[i].bbox.height
                    area[i] = w1 * h1

                    for j in xrange(i, len(boxes)):
                        x2 = boxes[j].bbox.x
                        y2 = boxes[j].bbox.y
                        w2 = boxes[j].bbox.width
                        h2 = boxes[j].bbox.height
                        intersection[i,j] = intersection[j,i] = get_overlap_area([x1, y1, w1, h1], [x2, y2, w2, h2])
                        
                sqrtArea = np.sqrt(np.array(area))
                sqrtArea = sqrtArea * sqrtArea.transpose()
                contain = (intersection / np.array([area]).transpose() >= 1).sum(axis = 1) - 1
                penalty = np.exp(-self.gamma * contain)

                quality_term = self.get_quality_term(weights, penalty)
                similarity_term = self.get_similarity_term(features, intersection, sqrtArea)
                return self.greedy_solve(boxes, quality_term, similarity_term)
            else:
                quality_term = np.empty(len(boxes), dtype = float)
                features = np.empty((len(boxes),boxes[0].feature.shape[0]))
                for idx, box in enumerate(boxes):
                    quality_term[idx] = box.conf
                    features[idx,:] = box.feature
                similarity_term = self.affinity_kernel(boxes, img_size = img_size)
                #similarity_term = np.dot(features, features.transpose())
                return self.greedy_solve(boxes, quality_term, similarity_term)
        return []

    def get_quality_term(self, weights, penalty):
        qt = weights * penalty
        qt = qt / qt.max() + 1
        qt = np.log10(qt)
        qt = self.alpha * qt * self.beta
        qt = np.square(qt)
        return qt

    def get_similarity_term(self, features, intersection, sqrtArea):
        Ss = intersection / sqrtArea
        Sc = np.matmul(features, features.transpose())
        return self.mu * Ss + (1 - self.mu) * Sc
    
    def greedy_solve(self, boxes, quality_term, similarity_term):
        boxes_tmp = boxes[:]
        boxes_list = [(i,box,True) for i, box in enumerate(boxes)]
        _lambda = 1.0
        quality_term = np.sqrt(_lambda * np.exp(quality_term))
        #quality_term = np.exp(- quality_term)
        argMax = quality_term.argmax()
        old_prob = quality_term[argMax]
        box_selected = boxes_list[argMax][1]
        boxes_list[argMax] = (argMax, box_selected, False)
        indices = np.array([argMax], dtype = int)
        while np.sum([ b for i,v,b in boxes_list]):
            prob = 0.0
            for i, box, b in boxes_list:
                if b:
                    tmp_indices = np.append(indices, i)
                    tmpProb = quality_term[i] * np.linalg.det(similarity_term[np.ix_(tmp_indices,tmp_indices)])
                    #tmpProb = (quality_term[tmp_indices]).prod() * np.linalg.det(similarity_term[np.ix_(tmp_indices,tmp_indices)])
                    #print 'tmpIndices:' + str(tmp_indices) + ' || tmpProb:' + str(tmpProb)+ ' - prod(qt^2):' + str((quality_term[tmp_indices]**2).prod()) + \
                    #' - det(S_y):' + str(np.linalg.det(similarity_term[np.ix_(tmp_indices,tmp_indices)]))
                    if tmpProb > prob:
                        argMax = i
                        prob = tmpProb
            prob = quality_term[indices].prod() * prob
            if float(prob)/old_prob > 1.0 + 0.1:
            #if np.log(prob) > np.log(old_prob):
                indices = np.append(indices, argMax)
                old_prob = prob
                box_selected = boxes_list[argMax][1]
                boxes_list[argMax] = (argMax, box_selected, False)
            else:
                break
        if type(indices) is np.ndarray:
            return [boxes[i] for i in indices]
        else:
            return []

    def affinity_kernel(self, tracks, w = 100.0, img_size = None):
        kernel = np.empty((len(tracks),len(tracks)), dtype = float)

        app = np.empty((len(tracks),len(tracks)), dtype = float)
        scale = np.empty((len(tracks),len(tracks)), dtype = float)
        position = np.empty((len(tracks),len(tracks)), dtype = float)
        (img_width, img_height) = img_size
        diag = np.sqrt( np.power(img_height, 2) + np.power(img_width, 2) )
        
        for i in xrange(len(tracks)):
            feat1 = tracks[i].feature
            x1 = tracks[i].bbox.x
            y1 = tracks[i].bbox.y
            w1 = tracks[i].bbox.width
            h1 = tracks[i].bbox.height
            #print str(x1) + ',' + str(y1) + ',' + str(w1) + ',' + str(h1)
            for j in xrange(i,len(tracks)):
                feat2 = tracks[j].feature
                x2 = tracks[j].bbox.x
                y2 = tracks[j].bbox.y
                w2 = tracks[j].bbox.width
                h2 = tracks[j].bbox.height
                #app[i,j] = app[j,i] = np.exp( -w * ((feat2-feat1)**2).sum())
                app[i,j] = app[j,i] = np.dot(feat1,feat2)/(np.linalg.norm(feat1) * np.linalg.norm(feat2))
                scale[i,j] = scale[j,i] = np.exp( -w * (np.sqrt(np.power(w2 - w1, 2) + np.power(h2 - h1, 2)))/diag )
                position[i,j] = position[j,i] = np.exp( -w * (np.sqrt(np.power(x2 - x1, 2) + np.power(y2 - y1, 2)))/diag )
                #scale[i,j] = np.exp( -w * ( ((math.fabs(w2 - w1))/(math.fabs(w2 + w1))) + ((math.fabs(h2 - h1))/(math.fabs(h2 + h1))) ) )
                #position[i,j] = np.exp( -w * (np.power( (x2 - x1)/(w2),2 ) + np.power((y2 -y1)/(h2), 2) ) )
                
        kernel = app * scale * position
        #print kernel.flatten()
        #exit()
        #kernel = app
        #1-app 2-scale-position
        return kernel