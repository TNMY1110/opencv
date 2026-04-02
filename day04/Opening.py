import urllib.request
import os
import numpy as np 
import cv2 as cv 

def get_sample(filename, repo='insightbook'):
    if not os.path.exists(filename):
        if repo == 'insightbook':
            url = f"https://raw.githubusercontent.com/dltpdn/insightbook.opencv_project_python/master/img/{filename}"
        else:  # opencv 공식
            url = f"https://raw.githubusercontent.com/opencv/opencv/master/samples/data/{filename}"
        urllib.request.urlretrieve(url, f"./img/{filename}")
    return f"./img/{filename}"

img = cv.imread(get_sample('4027.png'))
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# 이미지 반전 (흰색 배경 + 검정글씨 -> 검정 배경 + 흰색글씨 )
inverted = cv.bitwise_not(gray)
kernel = cv.getStructuringElement(cv.MORPH_CROSS, (5,5))

eroded = cv.erode(inverted, kernel, iterations=1)
dialed = cv.dilate(inverted, kernel, iterations=1)
opened = cv.morphologyEx(inverted, cv.MORPH_OPEN, kernel)

cv.imshow("Original", img)
cv.imshow("Inverted", inverted)
cv.imshow("Edosion", eroded)
cv.imshow("Dilation", dialed)
cv.imshow("opening", opened)

cv.waitKey(0)
cv.destroyAllWindows()
