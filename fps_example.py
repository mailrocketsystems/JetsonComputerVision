import cv2
import imutils
import datetime

cap = cv2.VideoCapture("videos/test_video.mp4")
fps_start_time = datetime.datetime.now()
fps = 0
total_frames = 0

while True:
    ret, frame = cap.read()
    frame = imutils.resize(frame, width=800)
    total_frames = total_frames + 1

    fps_end_time = datetime.datetime.now()
    time_diff = fps_end_time - fps_start_time
    if time_diff.seconds == 0:
        fps = 0.0
    else:
        fps = (total_frames / time_diff.seconds)

    fps_txt = "FPS: {:.2f}".format(fps)
    cv2.putText(frame, fps_txt, (5, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)

    cv2.imshow("Rocket System", frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cv2.destroyAllWindows()

    
