import numpy as np
import cv2 as cv
import urllib.request
import os

def get_sample(filename, repo='insightbook'):
    """부교 또는 OpenCV 공식 샘플 이미지 자동 다운로드
    
    Args:
        filename (str): 이미지 파일명 (예: 'morphological.png')
        repo (str): 'insightbook' (부교) 또는 'opencv' (공식)
    
    Returns:
        str: 다운로드된 파일명
    """
    if not os.path.exists(filename):
        if repo == 'insightbook':
            url = f"https://raw.githubusercontent.com/dltpdn/insightbook.opencv_project_python/master/img/{filename}"
        else:  # opencv 공식
            url = f"https://raw.githubusercontent.com/opencv/opencv/master/samples/data/{filename}"
        urllib.request.urlretrieve(url, filename)
    return filename

# 사용 방법
img = cv.imread(get_sample('messi5.jpg', repo='opencv'))

assert img is not None, "file could not be read, check with os.path.exists()"
 
res = cv.resize(img, None, fx=2, fy=2, interpolation = cv.INTER_CUBIC)

cv.imshow('Original', img)
cv.imshow('Scaling', res)

cv.waitKey(0)
cv.destroyAllWindows()
