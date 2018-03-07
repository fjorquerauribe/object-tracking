import mxnet as mx
import numpy as np
import cv2

from collections import namedtuple
Batch = namedtuple('Batch', ['data'])

class Resnet:
    sym = None
    arg_params = None
    aux_params = None
    DIM = 2048

    def __init__(self):
        self.sym, self.arg_params, self.aux_params = mx.model.load_checkpoint('../data/resnet-152', 0)
        mod = mx.mod.Module(symbol = self.sym, context = mx.gpu(), label_names = None)
        mod.bind(for_training = False, data_shapes = [('data', (1,3,224,224))], 
         label_shapes = mod._label_shapes)
        mod.set_params(self.arg_params, self.aux_params, allow_missing = True)
        DIM = 2048

    def get_subimage(self, img, box):
        if img is None:
            return None
        subimage = img[box.bbox.p_min[1]:box.bbox.p_max[1], box.bbox.p_min[0]:box.bbox.p_max[0]]
        subimage = cv2.cvtColor(subimage, cv2.COLOR_BGR2RGB)
        #print('img_shape:' + str(img.shape) + ' - box:' + str(box.bbox.p_min) + ',' + str(box.bbox.p_max))
        # convert into format (batch, RGB, width, height)
        subimage = cv2.resize(subimage, (224, 224))
        subimage = np.swapaxes(subimage, 0, 2)
        subimage = np.swapaxes(subimage, 1, 2)
        subimage = subimage[np.newaxis, :]
        return subimage

    def get_features(self, img, boxes):
        all_layers = self.sym.get_internals()
        all_layers.list_outputs()[-10:]

        fe_sym = all_layers['flatten0_output']
        fe_mod = mx.mod.Module(symbol = fe_sym, context = mx.gpu(), label_names = None)
        fe_mod.bind(for_training = False, data_shapes = [('data', (1,3,224,224))])
        fe_mod.set_params(self.arg_params, self.aux_params)

        features = np.empty((len(boxes), self.DIM))
        for i in xrange(len(boxes)):
            subimage = self.get_subimage(img, boxes[i])
            fe_mod.forward(Batch([mx.nd.array(subimage)]))
            features[i,:] = fe_mod.get_outputs()[0].asnumpy()
        return features

if __name__ == '__main__':
    detecton = Resnet()