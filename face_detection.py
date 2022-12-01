import cv2
import datetime
import imutils
import numpy as np 

protopath = "model/face/deploy.prototxt"
modelpath = "model/face/res10_300x300_ssd_iter_140000.caffemodel"
detector = cv2.dnn.readNetFromCaffe(protopath, modelpath)

cap = cv2.VideoCapture("videos/test_video.mp4")

while True:
    ret, frame = cap.read()
    frame = imutils.resize(frame, width=600)
    (H, W) = frame.shape[:2]
    face_blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0), False, False)
    detector.setInput(face_blob)
    face_detections = detector.forward()

    for i in np.arange(0, face_detections.shape[2]):
        confidence = face_detections[0, 0, i, 2]
        if confidence > 0.2:
            face_box = face_detections[0, 0, i, 3:7] * np.array([W, H, W, H])
            (startX, startY, endX, endY) = face_box.astype("int")

            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)

    cv2.imshow("Results", frame)
    key = cv2.waitKey(1)
    if key == ord("q"):
        break
cv2.destroyAllWindows()