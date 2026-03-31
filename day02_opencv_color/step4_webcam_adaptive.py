import cv2 as cv
import numpy as np
import urllib.request
import os
from matplotlib import pyplot as plt

cap = cv.VideoCapture(0)

def nothing(x):
    pass

cv.namedWindow('thr')

cv.createTrackbar('Block_Size', 'thr', 11, 31, nothing)
cv.createTrackbar('C', 'thr', 2, 20, nothing)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

while(1):
    # global thresholding
    ret, frame = cap.read()
    
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    blockSize = cv.getTrackbarPos('Block_Size', 'thr')
    blockSize = max(3, blockSize | 1)  # 비트 or 연산자로 짝수면 +1, 홀수면 놔두고 최솟값 3
    C = cv.getTrackbarPos('C', 'thr')

    _, th1 = cv.threshold(gray, 0, 255, cv.THRESH_BINARY)
    _, th2 = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    th3 = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, blockSize, C)
    th4 = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, blockSize, C)

    top  = np.hstack([th1, th2])
    bottom = np.hstack([th3, th4])
    result = np.vstack([top, bottom])
    cv.imshow('thr', result)

    k = cv.waitKey(5) & 0xFF

    if k == ord('q'):
        break

cv.destroyAllWindows()