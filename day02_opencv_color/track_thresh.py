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

img = get_sample('sudoku.png', cv.IMREAD_GRAYSCALE)
assert img is not None, "file could not be read, check with os.path.exists()"

while(1):
    # global thresholding
    Glo_thr = cv.getTrackbarPos('Global_thr', 'thr')
    ret, th1 = cv.threshold(img, Glo_thr, 255, cv.THRESH_BINARY)

    # Otsu's thresholding
    ret, th2 = cv.threshold(img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    th3 = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11,2)

    # plot all the images and their histograms
    row1 = np.hstack([img, th1])   # 위쪽 행
    row2 = np.hstack([th2, th3])   # 아래쪽 행
    result = np.vstack([row1, row2])  # 두 행을 세로로 합치기
    cv.imshow('thr', result)

    k = cv.waitKey(5) & 0xFF

    if k == 27:
        break

cv.destroyAllWindows()

