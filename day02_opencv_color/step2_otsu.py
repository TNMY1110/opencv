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
mode = '0 : OFF \n1 : ON'
cv.createTrackbar(mode, 'thr', 0, 1, nothing)

img = get_sample('sudoku.png', cv.IMREAD_GRAYSCALE)
assert img is not None, "file could not be read, check with os.path.exists()"
th1 = np.zeros_like(img)

while(1):
    # global thresholding
    Glo_thr = cv.getTrackbarPos('Global_thr', 'thr')
    mode_val = cv.getTrackbarPos('0 : OFF \n1 : ON', 'thr')

    _, th1 = cv.threshold(img, Glo_thr, 255, cv.THRESH_BINARY)

    ret, th2 = cv.threshold(img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    cv.putText(th2, f"Otsu: {int(ret)}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, 150, 2)

    if  mode_val == 0:
        result = np.hstack([img, th2])
        cv.imshow('thr', result)

    elif  mode_val == 1:
        result = np.hstack([img, th1, th2])
        cv.imshow('thr', result)

    k = cv.waitKey(5) & 0xFF

    if k == ord('q'):
        break

cv.destroyAllWindows()