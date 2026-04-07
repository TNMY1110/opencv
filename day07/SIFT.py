import numpy as np
import cv2 as cv
from sample_download import get_sample
 
img = cv.imread(f"./img/{get_sample('home.jpg')}")
gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)

sift = cv.SIFT_create()
kp = sift.detect(gray,None)
print(kp)

img=cv.drawKeypoints(gray,kp,img)
 
cv.imwrite('./img/sift_keypoints.jpg',img)