# import the necessary packages
from imutils.video import FileVideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2
import dlib
from imutils.video import VideoStream




# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()

ap.add_argument("-v", "--video", required=True,
                help="path to input video file")

ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")
#To create a Caffe model you need to define the model architecture in a protocol buffer definition file (prototxt).

#The model well be using in this blog post is a
#Caffe version of the original TensorFlow implementation by Howard et al. and was trained by chuanqi305

ap.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")
#The MobileNet SSD was first trained on the COCO dataset
# (Common Objects in Context) and was then fine-tuned on PASCAL VOC reaching 72.7% mAP (mean average precision)

ap.add_argument("-c", "--confidence", type=float, default=0.2,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())


CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
#mekedi wenne 0th(include) 255th(exclude)
#  athara random numbers thiyana array ekak hadana eka..meke size eka dila thiyenne ilakkam dekakin enisa 2d array ekak hadenne
#classes eke thyana awayawa ganata samata awayawa 3k athi tuples thiyana array ekak.awayawa 0-255 random floats
# load our serialized model from disk

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
# caffe= GoogLeNet trained network from Caffe model zoo.
#meke prototxt ekakui(architecture eka thiyenne meeke) model ekakui(meka train krla thyenne object detect krnna) denna oni

#meken return karanne net object ekak

# initialize the video stream, allow the cammera sensor to warmup,
# and initialize the FPS counter


# start the file video stream thread and allow the buffer to
# start to fill
print("[INFO] starting video file thread...")
fvs = FileVideoStream(args["video"]).start()
time.sleep(1.0)

# start the FPS timer
fps = FPS().start()
tracker=[]
while fvs.more():
    # grab the frame from the threaded video file stream, resize
    # it, and convert it to grayscale (while still retaining 3
    # channels)
    frame = fvs.read()
    frame = imutils.resize(frame, width=450)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    (h, w) = frame.shape[:2]

    # [Shape of image is accessed by img.shape. It returns a tuple of number
    # of rows, columns and channels (if image is color):

    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),  # convert frame into a blob
                                 0.007843, (300, 300), 127.5)

    net.setInput(blob)  # me blob eka nural network ekata deela eke detections gannawa
    detections = net.forward()  # forward method eken return wenne <type 'numpy.ndarray'>.
    coordinates = []


    frame = imutils.resize(frame, width=450)

    frame = np.dstack([frame, frame, frame])
    for i in np.arange(0, detections.shape[2]):#shape eken return wenne array eke dimention eka shape[2] kiyanne eke 3weni attribute
        #  eka detect kragatta objects gana
            # take the probability of each detection
        confidence = detections[0, 0, i, 2]
            # filter out weak detections by ensuring the `confidence` is
            # greater than the minimum confidence given
        if confidence > args["confidence"]:
    #
            idx = int(detections[0, 0, i, 1])#get the index from detected object
            # 1 wenne i kiyana object eke wargaya class eke position eka
            if( idx==15):  #filter only humans
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])##3:7 wenne i kiyana object eke x1=3,y1=4,x2=5,y2=6 points 4
                #print box.astype("int")
                (startX, startY, endX, endY) = box.astype("int")
                coordinates.append((startX, startY, endX, endY))
                    # draw the prediction on the frame
                label = "{}: {:.2f}%".format(CLASSES[idx],
                                                 confidence * 100)

                #cv2.rectangle(frame, (startX, startY), (endX, endY),(255,255,255), 2)
                tracker = dlib.correlation_tracker()
                tracker.start_track(frame, dlib.rectangle(startX,startY,endX,endY))
                # = [dlib.correlation_tracker() for _ in xrange(len(coordinates))]
                    # Provide the tracker the initial position of the object
        #[tracker[i].start_track(frame, dlib.rectangle(*rect)) for i, rect in enumerate(coordinates)]
    #for ii in xrange(len(tracker)):
                tracker.update(frame)
        # Get the position of th object, draw a
        # bounding box around it and display it.
                rect = tracker.get_position()
                pt1 = (int(rect.left()), int(rect.top()))
                pt2 = (int(rect.right()), int(rect.bottom()))
                cv2.rectangle(frame, pt1, pt2, (255, 255, 255), 3)

                label = str(i)
                # Get the position of the object, draw a
                # bounding box around it and display it.
                y = pt1[1] - 15 if pt1[1] - 15 > 15 else pt1[1] + 15
                cv2.putText(frame, label, (pt1[0], y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

    # display the size of the queue on the frame
    cv2.putText(frame, "Queue Size: {}".format(fvs.Q.qsize()),
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # show the frame and update the FPS counter
    cv2.imshow("Frame", frame)
    cv2.waitKey(1)
    fps.update()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
fvs.stop()