import numpy as np
import cv2 as cv
import urllib.request
import os
import matplotlib.pylab as plt 

def get_sample(filename, repo='insightbook'):
    if not os.path.exists(filename):
        if repo == 'insightbook':
            url = f"https://raw.githubusercontent.com/dltpdn/insightbook.opencv_project_python/master/img/{filename}"
        else:  # opencv 공식
            url = f"https://raw.githubusercontent.com/opencv/opencv/master/samples/data/{filename}"
        urllib.request.urlretrieve(url, filename)
    return filename

# 사용 방법
img = cv.imread(get_sample('opencv-logo-white.png', repo='opencv'))
assert img is not None, "file could not be read, check with os.path.exists()"

blur = cv.blur(img,(5,5))
Gausblur = cv.GaussianBlur(img,(5,5),0)
median = cv.medianBlur(img,5)
bilFilter = cv.bilateralFilter(img,9,75,75)

titles = ['Original', 'Blurred', 'Gaussian Blurred', 'Median Blurred']
images = [img, blur, Gausblur, median]

for i in range(4):
    plt.subplot(2, 2, i+1)  # 2행 2열, i+1번째 위치
    plt.imshow(images[i])
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])

plt.tight_layout()
plt.show()

cv.waitKey(0)
cv.destroyAllWindows()