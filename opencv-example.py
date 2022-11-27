import cv2
import imutils

# This is how we load frames from a video file
cap = cv2.VideoCapture("videos/test_video.mp4")
while True:
    ret, frame = cap.read()
    frame = imutils.resize(frame, width=800)
    cv2.putText(frame, "This is my text", (20, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
    cv2.rectangle(frame, (30, 30), (400, 400), (0, 0, 255), 2)
    cv2.imshow("Rocket Systems", frame)
    key = cv2.waitKey(1)
    if key == ord("q"):
        break
cv2.destroyAllWindows()


"""
# This is how we read image files.
image = cv2.imread("img/img.jpg")
image = imutils.resize(image, width=800)
cv2.putText(image, "This is my text", (20, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
cv2.rectangle(image, (30, 30), (400, 400), (0, 0, 255), 2)
cv2.imshow("Rocket Systems", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""