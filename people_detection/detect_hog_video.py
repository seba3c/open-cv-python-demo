import cv2

import numpy as np

import imutils
from imutils.object_detection import non_max_suppression

cap = cv2.VideoCapture('time_square.mp4')

# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

winStride = (4, 4)
padding = (4, 4)
scale = 1.05
rect_color = (0, 255, 0)

frame_count = 0

cv2.namedWindow("PeopleDetection", cv2.WINDOW_GUI_EXPANDED)

while(True):
    ret, frame = cap.read()

    frame_count += 1

    if (frame_count % 3) != 0:
        continue

    frame = imutils.resize(frame, width=min(400, frame.shape[1]))

    (rects, weights) = hog.detectMultiScale(frame, winStride=winStride, padding=padding, scale=scale)

    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
    # pick = rects

    # draw the final bounding boxes
    for (xA, yA, xB, yB) in pick:
        cv2.rectangle(frame, (xA, yA), (xB, yB), rect_color, 2)

    cv2.imshow('PeopleDetection', frame)

    key = cv2.waitKey(30) & 0xff
    # key = cv2.waitKey(0)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
