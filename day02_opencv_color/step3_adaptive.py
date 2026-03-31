import cv2 as cv
import numpy as np
import urllib.request
import os
from matplotlib import pyplot as plt

def get_sample(filename, flags=cv.IMREAD_COLOR):
    if not os.path.exists(filename):
        url = f"https://raw.githubusercontent.com/opencv/opencv/master/samples/data/{filename}"
        urllib.request.urlretrieve(url, filename)
    
    return cv.imread(filename, flags)

def nothing(x):
    pass

cv.namedWindow('thr', cv.WINDOW_AUTOSIZE)
cv.createTrackbar('Global_thr', 'thr', 127, 255, nothing)
cv.createTrackbar('Block_Size', 'thr', 11, 31, nothing)
cv.createTrackbar('C', 'thr', 2, 20, nothing)
mode = '0 : OFF \n1 : ON'
cv.createTrackbar(mode, 'thr', 0, 1, nothing)

img = get_sample('sudoku.png', cv.IMREAD_GRAYSCALE)
assert img is not None, "file could not be read, check with os.path.exists()"
th1 = np.zeros_like(img)

while(1):
    # global thresholding
    Glo_thr = cv.getTrackbarPos('Global_thr', 'thr')
    mode_val = cv.getTrackbarPos('0 : OFF \n1 : ON', 'thr')
    blockSize = cv.getTrackbarPos('Block_Size', 'thr')
    blockSize = max(3, blockSize | 1)  # 비트 or 연산자로 짝수면 +1, 홀수면 놔두고 최솟값 3
    C = cv.getTrackbarPos('C', 'thr')

    if  mode_val == 0:
        _, th1 = cv.threshold(img, Glo_thr, 255, cv.THRESH_BINARY)

    elif  mode_val == 1:
        _, th1 = cv.threshold(img, Glo_thr, 255, cv.THRESH_BINARY_INV)

    ret, th2 = cv.threshold(img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    cv.putText(th2, f"Otsu: {int(ret)}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, 150, 2)

    # Adaptive Mean Threshold
    th3 = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, blockSize, C)
    th4 = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, blockSize, C)

    top  = np.hstack([th1, th2])
    bottom = np.hstack([th3, th4])
    result = np.vstack([top, bottom])
    cv.imshow('thr', result)

    k = cv.waitKey(5) & 0xFF

    if k == ord('q'):
        break

cv.destroyAllWindows()