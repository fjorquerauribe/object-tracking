import argparse
import os
import cv2
import mxnet as mx
import numpy as np
from utils import Rectangle, Detection
from rcnn.logger import logger
from rcnn.config import config
from rcnn.symbol import get_vgg_test, get_vgg_rpn_test
from rcnn.io.image import resize, transform
from rcnn.core.tester import Predictor, im_detect, im_proposal, vis_all_detection, draw_all_detection
from rcnn.utils.load_model import load_param
from rcnn.processing.nms import py_nms_wrapper, cpu_nms_wrapper, gpu_nms_wrapper

CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')
#CLASSES = ('__background__', 'person')
config.TEST.HAS_RPN = True
SHORT_SIDE = config.SCALES[0][0]
LONG_SIDE = config.SCALES[0][1]
PIXEL_MEANS = config.PIXEL_MEANS
DATA_NAMES = ['data', 'im_info']
LABEL_NAMES = None
DATA_SHAPES = [('data', (1, 3, LONG_SIDE, SHORT_SIDE)), ('im_info', (1, 3))]
LABEL_SHAPES = None

CONF_THRESH = 0.7 #0.7
NMS_THRESH = 0.3 #0.3
nms = py_nms_wrapper(NMS_THRESH)

class FasterRCNN:
    def __init__(self, gpu = 0, prefix = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'final')), epoch = 0):
        ctx = mx.gpu(gpu)
        symbol = get_vgg_test(num_classes = config.NUM_CLASSES, num_anchors = config.NUM_ANCHORS)
        self.predictor = self.get_net(symbol, prefix, epoch, ctx)

    def get_net(self, symbol, prefix, epoch, ctx):
        arg_params, aux_params = load_param(prefix, epoch, convert=True, ctx=ctx, process=True)

        # infer shape
        data_shape_dict = dict(DATA_SHAPES)
        arg_names, aux_names = symbol.list_arguments(), symbol.list_auxiliary_states()
        arg_shape, _, aux_shape = symbol.infer_shape(**data_shape_dict)
        arg_shape_dict = dict(zip(arg_names, arg_shape))
        aux_shape_dict = dict(zip(aux_names, aux_shape))

        # check shapes
        for k in symbol.list_arguments():
            if k in data_shape_dict or 'label' in k:
                continue
            assert k in arg_params, k + ' not initialized'
            assert arg_params[k].shape == arg_shape_dict[k], \
                'shape inconsistent for ' + k + ' inferred ' + str(arg_shape_dict[k]) + ' provided ' + str(arg_params[k].shape)
        for k in symbol.list_auxiliary_states():
            assert k in aux_params, k + ' not initialized'
            assert aux_params[k].shape == aux_shape_dict[k], \
                'shape inconsistent for ' + k + ' inferred ' + str(aux_shape_dict[k]) + ' provided ' + str(aux_params[k].shape)

        predictor = Predictor(symbol, DATA_NAMES, LABEL_NAMES, context = ctx,
                            provide_data = DATA_SHAPES, provide_label = LABEL_SHAPES,
                            arg_params = arg_params, aux_params = aux_params)
        return predictor

    def generate_batch(self, im):
        """
        preprocess image, return batch
        :param im: cv2.imread returns [height, width, channel] in BGR
        :return:
        data_batch: MXNet input batch
        data_names: names in data_batch
        im_scale: float number
        """
        im_array, im_scale = resize(im, SHORT_SIDE, LONG_SIDE)
        im_array = transform(im_array, PIXEL_MEANS)
        im_info = np.array([[im_array.shape[2], im_array.shape[3], im_scale]], dtype = np.float32)
        data = [mx.nd.array(im_array), mx.nd.array(im_info)]
        data_shapes = [('data', im_array.shape), ('im_info', im_info.shape)]
        data_batch = mx.io.DataBatch(data = data, label = None, provide_data = data_shapes, provide_label = None)
        return data_batch, DATA_NAMES, im_scale

    def detect(self, im, vis = False, verbose = False):
        """
        generate data_batch -> im_detect -> post process
        :param vis: will save as a new image if not visualized
        :return: None
        """
        data_batch, data_names, im_scale = self.generate_batch(im)
        scores, boxes, data_dict = im_detect(self.predictor, data_batch, data_names, im_scale)
        
        all_boxes = [[] for _ in CLASSES]
        for cls in CLASSES:
            #if cls == 'person':
            cls_ind = CLASSES.index(cls)
            cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
            cls_scores = scores[:, cls_ind, np.newaxis]
            keep = np.where(cls_scores >= CONF_THRESH)[0]
            dets = np.hstack((cls_boxes, cls_scores)).astype(np.float32)[keep, :]
            keep = nms(dets)
            all_boxes[cls_ind] = dets[keep, :]
        boxes_this_image = [[]] + [all_boxes[j] for j in range(1, len(CLASSES))]

        detections = []
        for ind, boxes in enumerate(boxes_this_image):
            if len(boxes) > 0:
                for box in boxes:
                    x1 = int(round(box[0])) 
                    y1 = int(round(box[1]))
                    x2 = int(round(box[2]))
                    y2 = int(round(box[3]))
                    score = int(round(box[4]))
                    new_det = Detection(x1, y1, x2 - x1, y2 - y1, score)
                    detections.append(new_det)
                    if verbose:
                        format_str = ('%d,%d,%d,%d,%.3f')
                        print(format_str % (x1,y1,x2-x1,y2-y1,score))
        return detections