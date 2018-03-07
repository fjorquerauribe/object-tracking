import numpy as np
from utils import get_overlap_area, get_overlap_ratio

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

    def run(self, detections, weights, features):
        if len(detections) > 0:
            area = np.empty(len(detections), dtype = float)
            intersection = np.empty((len(detections), len(detections)) , dtype = float)

            for i in xrange(len(detections)):
                x1 = detections[i].bbox.p_min[0]
                y1 = detections[i].bbox.p_min[1]
                w1 = detections[i].bbox.p_max[0] - detections[i].bbox.p_min[0]
                h1 = detections[i].bbox.p_max[1] - detections[i].bbox.p_min[1]
                area[i] = w1 * h1

                for j in xrange(i, len(detections)):
                    x2 = detections[j].bbox.p_min[0]
                    y2 = detections[j].bbox.p_min[1]
                    w2 = detections[j].bbox.p_max[0] - detections[j].bbox.p_min[0]
                    h2 = detections[j].bbox.p_max[1] - detections[j].bbox.p_min[1]

                    intersection[i,j] = intersection[j,i] = get_overlap_area([x1, y1, w1, h1], [x2, y2, w2, h2])
                    
            sqrtArea = np.sqrt(np.array(area))
            sqrtArea = sqrtArea * sqrtArea.transpose()
            contain = (intersection / np.array([area]).transpose() >= 1).sum(axis = 1) - 1
            penalty = np.exp(-self.gamma * contain)

            quality_term = self.get_quality_term(weights, penalty)
            similarity_term = self.get_similarity_term(features, intersection, sqrtArea)
            return self.greedy_solve(detections, quality_term, similarity_term)
        return np.array([])

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

    def greedy_solve(self, detections, quality_term, similarity_term):
        dets = detections[:]
        
        prob = quality_term * np.diag(similarity_term)
        #print 'diag: ' + str(np.diag(similarity_term))
        argMax = prob.argmax()
        prob = prob[argMax]
        
        indices = np.array([argMax], dtype = int)
        del dets[argMax]

        while dets:
            old_prob = prob
            for i in xrange(len(dets)):
                tmpProb = quality_term[np.append(indices,i)].prod() * np.linalg.det(similarity_term[np.ix_(np.append(indices,i),np.append(indices,i))])
                
                #print 'qt: ' + str(quality_term[np.append(indices,i)].prod()) + ' | det: ' + str(np.linalg.det(similarity_term[np.ix_(np.append(indices,i),np.append(indices,i))]))
                
                if tmpProb > prob:
                    argMax = i
                    prob = tmpProb
            #print float(prob)/old_prob
            #print str(float(prob)/old_prob) + " | " + str(1 + self.epsilon)  
            if float(prob)/old_prob > 1 + self.epsilon:
                indices = np.append(indices, argMax)
                del dets[argMax]
            else:
                break
        
        if type(indices) is np.ndarray:
            return indices
            #return [detections[i] for i in indices]
        else:
            return np.array([])