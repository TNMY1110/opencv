import urllib.request
import os
import cv2 as cv 
from matplotlib import pyplot as plt 
import numpy as np

def get_sample(filename, repo='opencv'):
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
# img = cv.imread(get_sample('mario.png', repo='insightbook'))

img_rgb = cv.imread('./img/mario.png')
#print(img_rgb)
assert img_rgb is not None, "file could not be read, check with os.path.exists()"
img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)
template = cv.imread('./img/coin.png', cv.IMREAD_GRAYSCALE)
print(template)
assert template is not None, "file could not be read, check with os.path.exists()"
w, h = template.shape[::-1]

res = cv.matchTemplate(img_gray,template,cv.TM_CCOEFF_NORMED)
threshold = 0.8
loc = np.where( res >= threshold)
for pt in zip(*loc[::-1]):
    cv.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)

cv.imwrite('./img/res.png',img_rgb)