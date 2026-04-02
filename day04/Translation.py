import numpy as np
import cv2 as cv
import urllib.request
import os

def get_sample(filename, repo='insightbook'):
    if not os.path.exists(filename):
        if repo == 'insightbook':
            url = f"https://raw.githubusercontent.com/dltpdn/insightbook.opencv_project_python/master/img/{filename}"
        else:  # opencv 공식
            url = f"https://raw.githubusercontent.com/opencv/opencv/master/samples/data/{filename}"
        urllib.request.urlretrieve(url, filename)
    return filename

# 사용 방법
img = cv.imread(get_sample('messi5.jpg', repo='opencv'), cv.IMREAD_GRAYSCALE)
assert img is not None, "file could not be read, check with os.path.exists()"
rows,cols = img.shape
 
M = np.float32([[1,0,100],[0,1,50]])
dst = cv.warpAffine(img,M,(cols,rows))
 
cv.imshow('img',dst)
cv.waitKey(0)
cv.destroyAllWindows()