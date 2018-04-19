# USAGE
# python test_pi_tracking.py --prototxt ../data/MobileNetSSD_deploy.prototxt.txt --model ../data/MobileNetSSD_deploy.caffemodel
# A script based on https://www.pyimagesearch.com/2017/10/16/raspberry-pi-deep-learning-object-detection-with-opencv/

# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
from multiprocessing import Process
from multiprocessing import Queue
import numpy as np
import argparse
import imutils
import time
import cv2

from context import tracking
from tracking.models.gmphd_filter import GMPHDFilter
from tracking.utils.utils import Detection

def classify_frame(net, inputQueue, outputQueue):
	while True:
		if not inputQueue.empty():
			frame = inputQueue.get()
			frame = cv2.resize(frame, (300, 300))
			blob = cv2.dnn.blobFromImage(frame, 0.007843,
				(300, 300), 127.5)

			net.setInput(blob)
			detections = net.forward()

			outputQueue.put(detections)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--prototxt", required=True,
        help="path to Caffe 'deploy' prototxt file")
    ap.add_argument("-m", "--model", required=True,
        help="path to Caffe pre-trained model")
    ap.add_argument("-c", "--confidence", type=float, default=0.2,
        help="minimum probability to filter weak detections")
    args = vars(ap.parse_args())

    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
        "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
        "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
        "sofa", "train", "tvmonitor"]
    COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

    print("[INFO] loading model...")
    net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

    verbose = False
    draw = True
    filter = GMPHDFilter(verbose)

    inputQueue = Queue(maxsize=1)
    outputQueue = Queue(maxsize=1)
    detections = None

    print("[INFO] starting process...")
    p = Process(target=classify_frame, args=(net, inputQueue,
        outputQueue,))
    p.daemon = True
    p.start()

    print("[INFO] starting video stream...")
    vs = VideoStream(src=0).start()
    # vs = VideoStream(usePiCamera=True).start()
    time.sleep(2.0)
    fps = FPS().start()

    while True:
        frame = vs.read()
        frame = imutils.resize(frame, width=400)
        (frame_height, frame_width, n_channels) = frame.shape
        (fH, fW) = frame.shape[:2]

        if inputQueue.empty():
            inputQueue.put(frame)

        if not outputQueue.empty():
            detections = outputQueue.get()

        if detections is not None:
            box_detections = []
            for i in np.arange(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]

                if confidence < args["confidence"]:
                    continue

                idx = int(detections[0, 0, i, 1])
                dims = np.array([fW, fH, fW, fH])
                box = detections[0, 0, i, 3:7] * dims
                (startX, startY, endX, endY) = box.astype("int")

                '''
                label = "{}: {:.2f}%".format(CLASSES[idx],  qq
                    confidence * 100)
                cv2.rectangle(frame, (startX, startY), (endX, endY),
                    COLORS[idx], 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(frame, label, (startX, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
                '''
                x = max(0,startX)
                y = max(0,startY)
                width = endX - x
                height = endY - y
                box_detections.append(Detection(x, y, width, height))
                #box_detections.append(Detection(max(0,startX), max(0,startY), min(frame_width,endX), min(frame_height,endY), confidence))
                
            if not filter.is_initialized():
                filter.initialize(frame, box_detections)
                estimates = filter.estimate(frame, draw = draw)
            else:
                filter.predict()
                filter.update(frame, box_detections, verbose = verbose)
                estimates = filter.estimate(frame, draw = draw)
                
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        fps.update()

    fps.stop()
    print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

    cv2.destroyAllWindows()
    vs.stop()
