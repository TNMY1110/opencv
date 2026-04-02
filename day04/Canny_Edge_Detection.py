import urllib.request
import os
import numpy as np 
import cv2 as cv 

def get_sample(filename, repo='opencv'):
    if not os.path.exists(filename):
        if repo == 'insightbook':
            url = f"https://raw.githubusercontent.com/dltpdn/insightbook.opencv_project_python/master/img/{filename}"
        else:  # opencv 공식
            url = f"https://raw.githubusercontent.com/opencv/opencv/master/samples/data/{filename}"
        urllib.request.urlretrieve(url, f"./img/{filename}")
    return f"./img/{filename}"

img = cv.imread(get_sample('messi5.jpg'))
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

blurred = cv.GaussianBlur(gray, (5,5), 1.5)

# 현재
edges_1 = cv.Canny(blurred, 100, 200, 4)
# 임계값 낮추기 -> 약한 에지도 검출 
edges_2 = cv.Canny(blurred, 90, 150, 4)
# 임계값  -> 약한 에지도 검출 
edges_3 = cv.Canny(blurred, 100, 300, 4)

cv.imshow("Original", img)
cv.imshow("Canny Edges (100, 200)_1", edges_1)
cv.imshow("Canny Edges (100, 200)_2", edges_2)
cv.imshow("Canny Edges (100, 200)_3", edges_3)

cv.waitKey(0)
cv.destroyAllWindows()
