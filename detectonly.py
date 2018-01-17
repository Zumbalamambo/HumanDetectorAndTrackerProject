# import the necessary packages
import dlib
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2

def remove_tracker(track_index):
    if track_index < len(tracker):
        tracker.remove(tracker[track_index])

def add_Tracker(frame,startX, startY, endX, endY):
    tt=dlib.correlation_tracker()
    tt.start_track(frame, dlib.rectangle(startX, startY, endX, endY))

    tracker.append(tt)
def is_olap(k,l):
    hoverlaps = True
    voverlaps = True

    if (k[0] > l[2]) or (k[2] < l[0]):
        hoverlaps = False
    if (k[3] < l[1]) or (k[1] > l[3]):
        voverlaps = False
    if (hoverlaps and voverlaps):
        return True
def overlaps(trcks,coordinates,frame):
    percentages=[]
    if len(trcks)>len(coordinates):
        #"a person leaves"
        for k in trcks:
            not_matching = True
            for l in coordinates:
                if is_olap(k, l):
                    not_matching = False
                    area1 = abs(k[0] - k[2]) * abs(k[1] - k[3])

                    cod1 = abs(l[0] - l[2]) * abs(l[1] - l[3])

                    is1 = abs(max(k[0], l[0]) - min(k[2], l[2])) * abs(
                        max(k[1], l[1]) - min(k[3], l[3]))
                    perc1 = (float(is1) / (area1 + cod1 - is1)) * 100
                    # if perc1>75:
                    #   print "success",k,l
                    percentages.append(round(perc1, 2))
                    print "overlaping percentage", perc1, k, l
            if not_matching:
                remove_tracker(trcks.index(k))
                print "this tracker should remove", k
    else:
        #"new person enterd"
        for k in coordinates:
            not_matching = True
            for l in trcks:
                if is_olap(k, l):
                    not_matching = False
                    area1 = abs(k[0] - k[2]) * abs(k[1] - k[3])

                    cod1 = abs(l[0] - l[2]) * abs(l[1] - l[3])

                    is1 = abs(max(k[0], l[0]) - min(k[2], l[2])) * abs(
                        max(k[1], l[1]) - min(k[3], l[3]))
                    perc1 = (float(is1) / (area1 + cod1 - is1)) * 100
                    # if perc1>75:
                    #   print "success",k,l
                    percentages.append(round(perc1, 2))
                    print "overlaping percentage", perc1, k, l
            if not_matching:
                add_Tracker(frame,k[0],k[1],k[2],k[3])
                print "new person came at", k



    #print percentages

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()

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

# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
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
print("[INFO] starting video stream...")
vs = cv2.VideoCapture("g.mp4")
time.sleep(2.0)
fps = FPS().start()
op=True
tracker=[]
# loop over the frames from the video stream
while True:
    # take frame by frame from video
    # change the width
    ret, frame = vs.read()

    #frame = imutils.resize(frame, width=600)


    (h, w) = frame.shape[:2]

    # [Shape of image is accessed by img.shape. It returns a tuple of number
    # of rows, columns and channels (if image is color):

    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),#convert frame into a blob
                                     0.007843, (300, 300), 127.5)


    net.setInput(blob)#me blob eka nural network ekata deela eke detections gannawa
    detections = net.forward()#forward method eken return wenne <type 'numpy.ndarray'>.

    coordinates=[]
    print "detections"
        # loop over the detections
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


                (startX, startY, endX, endY) = box.astype("int")
                #print [startX, startY, endX, endY],
                coordinates.append([startX, startY, endX, endY])
                    # draw the prediction on the frame
                #label = "{}: {:.2f}%".format(CLASSES[idx],
                                                 #confidence * 100)'
                label="person"

                cv2.rectangle(frame, (startX, startY), (endX, endY),
                                  COLORS[idx], 2)
                yyy = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(frame, label, (startX, yyy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

    print coordinates
    if(op):
        op=False
        tracker = [dlib.correlation_tracker() for _ in xrange(len(coordinates))]
                    # Provide the tracker the initial position of the object
        [tracker[i].start_track(frame, dlib.rectangle(*rect)) for i, rect in enumerate(coordinates)]
                    #tracker = dlib.correlation_tracker()
                    #tracker.start_track(frame, dlib.rectangle(startX,startY,endX,endY))
                # = [dlib.correlation_tracker() for _ in xrange(len(coordinates))]
                    # Provide the tracker the initial position of the object
        #[tracker[i].start_track(frame, dlib.rectangle(*rect)) for i, rect in enumerate(coordinates)]
    print
    print "trackers"
    trcks = []
    for ii in xrange(len(tracker)):
        tracker[ii].update(frame)
        # Get the position of th object, draw a
        # bounding box around it and display it.
        rect = tracker[ii].get_position()
        pt1 = (int(rect.left()), int(rect.top()))
        pt2 = (int(rect.right()), int(rect.bottom()))
        cv2.rectangle(frame, pt1, pt2, (255, 255, 255), 3)
        #print [int(rect.left()), int(rect.top()),int(rect.right()), int(rect.bottom())],
        trcks.append([int(rect.left()), int(rect.top()),int(rect.right()), int(rect.bottom())])
        label = "tl"+str(ii)
                # Get the position of the object, draw a
                # bounding box around it and display it.
        y = pt1[1] - 15 if pt1[1] - 15 > 15 else pt1[1] + 15
        cv2.putText(frame, label, (pt1[0], y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
    print trcks
    #SI = max(0, min(XA2, XB2) - Max(XA1, XB1)) * Max(0, Min(YA2, YB2) - Max(YA1, YB1))

    overlaps(trcks, coordinates,frame)
    print
    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

    # update the FPS counter
    fps.update()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()