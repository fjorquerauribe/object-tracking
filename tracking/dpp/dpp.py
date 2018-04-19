import numpy as np
import math
from utils import get_overlap_area, get_overlap_ratio, appearance_affinity, motion_affinity, shape_affinity

class DPP:
    alpha = 0.9
    beta = 1.1
    gamma = 0.1
    mu = 0.8
    epsilon = 0.1

    def __init__(self, alpha = 0.9, beta = 1.1, gamma = 0.1, mu = 0.8, epsilon = 0.1):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.mu = mu
        self.epsilon = epsilon

    def run(self, boxes = None, weights = None, features = None):
        if (len(boxes) > 0) and (boxes is not None):
            if (weights is not None) and (features is not None):
                    area = np.empty(len(boxes), dtype = float)
                    intersection = np.empty((len(boxes), len(boxes)) , dtype = float)
                    for i in xrange(len(boxes)):
                        x1 = boxes[i].bbox.p_min[0]
                        y1 = boxes[i].bbox.p_min[1]
                        w1 = boxes[i].bbox.p_max[0] - boxes[i].bbox.p_min[0]
                        h1 = boxes[i].bbox.p_max[1] - boxes[i].bbox.p_min[1]
                        area[i] = w1 * h1

                        for j in xrange(i, len(boxes)):
                            x2 = boxes[j].bbox.p_min[0]
                            y2 = boxes[j].bbox.p_min[1]
                            w2 = boxes[j].bbox.p_max[0] - boxes[j].bbox.p_min[0]
                            h2 = boxes[j].bbox.p_max[1] - boxes[j].bbox.p_min[1]
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
                similarity_term = self.affinity_kernel(boxes)
                
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
        _lambda = 0.1
        quality_term = np.sqrt(_lambda * np.exp(quality_term))
        prob = quality_term**2
        argMax = prob.argmax()
        prob = prob[argMax]
        old_prob = prob
        box_selected = boxes_list[argMax][1]
        boxes_list[argMax] = (argMax, box_selected, False)
        indices = np.array([argMax], dtype = int)
        #print 'prob: ' + str(old_prob)
        while np.sum([ b for i,v,b in boxes_list]):
            prob = 0.0
            for i, box, b in boxes_list:
                if b:
                    tmp_indices = np.append(indices, i)
                    tmpProb = (quality_term[tmp_indices]**2).prod() * np.linalg.det(similarity_term[np.ix_(tmp_indices,tmp_indices)])
                    #print 'tmpIndices:' + str(tmp_indices) + ' || tmpProb:' + str(tmpProb)+ ' - prod(qt^2):' + str((quality_term[tmp_indices]**2).prod()) + \
                    #' - det(S_y):' + str(np.linalg.det(similarity_term[np.ix_(tmp_indices,tmp_indices)]))
                    if tmpProb > prob:
                        argMax = i
                        prob = tmpProb
            #if float(prob)/old_prob > 1.0 + 0.1:
            if np.log(prob) < np.log(old_prob):
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
    '''
    def greedy_solve(self, boxes, quality_term, similarity_term):
        boxes_tmp = boxes[:]
        prob = (quality_term**2) #* np.diag(similarity_term)
        argMax = prob.argmax()
        prob = prob[argMax]
        indices = np.array([argMax], dtype = int)
        del boxes_tmp[argMax]
        while boxes_tmp:
            old_prob = prob
            prob = 0.0
            #old_prob = (quality_term[indices]**2).prod() * np.linalg.det(similarity_term[np.ix_(indices, indices)])
            for i in xrange(len(boxes_tmp)):
                tmpProb = (quality_term[np.append(indices,i)]**2).prod() * np.linalg.det(similarity_term[np.ix_(np.append(indices,i),np.append(indices,i))])
                if tmpProb > prob:
                    argMax = i
                    prob = tmpProb
            print float(prob)/old_prob
            if float(prob)/old_prob > 1.0 + self.epsilon:
                indices = np.append(indices, argMax)
                del boxes_tmp[argMax]
            else:
                break
        print len(indices)
        if type(indices) is np.ndarray:
            return [boxes[i] for i in indices]
        else:
            return []
    '''

    def affinity_kernel(self, tracks, w = 10.0):
        kernel = np.empty((len(tracks),len(tracks)), dtype = float)

        for i in xrange(len(tracks)):
            feat1 = tracks[i].feature
            x1 = tracks[i].bbox.p_min[0]
            y1 = tracks[i].bbox.p_min[1]
            w1 = tracks[i].bbox.p_max[0] - tracks[i].bbox.p_min[0]
            h1 = tracks[i].bbox.p_max[1] - tracks[i].bbox.p_min[1]
            
            for j in xrange(i, len(tracks)):
                feat2 = tracks[j].feature
                x2 = tracks[j].bbox.p_min[0]
                y2 = tracks[j].bbox.p_min[1]
                w2 = tracks[j].bbox.p_max[0] - tracks[j].bbox.p_min[0]
                h2 = tracks[j].bbox.p_max[1] - tracks[j].bbox.p_min[1]
                
                app = np.exp( -w * ((feat1 - feat2)**2).sum() )
                #motion = np.exp( -w * ( np.power( float(x1-x2)/float() ,2) + np.power( ,2) ) )
                print 'horizontal: ' + str(x1) + ' - ' + str(w1) + ' - ' + str(x2) + ' - ' + str(w2)
                print 'vertical: ' + str(y1) + ' - ' + str(h1) + ' - ' + str(y2) + ' - ' + str(h2)
                shape = np.exp( -w * (math.fabs(h1 - h2)/math.fabs(h1 + h2) + math.fabs(w1 - w2)/math.fabs(w1 + w2)))
                #position = 

                kernel[i,j] = app
                kernel[j,i] = app
                
        return kernel