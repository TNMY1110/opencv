import numpy as np
import cv2 as cv

cap = cv.VideoCapture(0)

def nothing(x):
    pass

cv.namedWindow('res')

cv.createTrackbar('Lower_H', 'res', 0, 179, nothing)
cv.createTrackbar('Lower_S', 'res', 0, 255, nothing)
cv.createTrackbar('Lower_V', 'res', 0, 255, nothing)

cv.createTrackbar('Upper_H', 'res', 179, 179, nothing)
cv.createTrackbar('Upper_S', 'res', 255, 255, nothing)
cv.createTrackbar('Upper_V', 'res', 255, 255, nothing)

while(1):
    _, frame = cap.read()
    
    lh = cv.getTrackbarPos('Lower_H', 'res')
    uh = cv.getTrackbarPos('Upper_H', 'res')

    ls = cv.getTrackbarPos('Lower_S', 'res')
    us = cv.getTrackbarPos('Upper_S', 'res')

    lv = cv.getTrackbarPos('Lower_V', 'res')
    uv = cv.getTrackbarPos('Upper_V', 'res')

    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    lower = np.array([lh, ls, lv])
    upper = np.array([uh, us, uv])

    # Threshold the HSV image to get only blue colors
    mask = cv.inRange(hsv, lower, upper)

    # Bitwise-AND mask and original image
    res = cv.bitwise_and(frame,frame, mask= mask)

    cv.imshow('res',res)

    k = cv.waitKey(5) & 0xFF
    # 27 == esc
    if k == 27:
        break

cv.destroyAllWindows()