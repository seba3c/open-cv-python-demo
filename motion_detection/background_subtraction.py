import cv2
import time
import numpy as np

# cap = cv2.VideoCapture('sample2.mp4')

cap = cv2.VideoCapture(0)

fgbg_MOG2 = cv2.createBackgroundSubtractorMOG2()
fgbg_KNN = cv2.createBackgroundSubtractorKNN()
fgbg_MOG = cv2.bgsegm.createBackgroundSubtractorMOG()
fgbg_GMG = cv2.bgsegm.createBackgroundSubtractorGMG()

firstFrame = None

while(1):
    ret, frame = cap.read()

    if firstFrame is None:
        firstFrame = frame
        cv2.imshow('Fondo', firstFrame)

    fgmask_mog2 = fgbg_MOG2.apply(frame)
    fgmask_knn = fgbg_KNN.apply(frame)
    fgmask_mog = fgbg_MOG.apply(frame)
    fgmask_gmg = fgbg_GMG.apply(frame)

    cv2.imshow('Video', frame)
    cv2.imshow('MOG 2', fgmask_mog2)
    cv2.imshow('KNN', fgmask_knn)
    cv2.imshow('MOG', fgmask_mog)
    cv2.imshow('GMG', fgmask_gmg)

    key = cv2.waitKey(30) & 0xff
    # key = cv2.waitKey(0)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
