import cv2 as cv 
import urllib.request
import sys 
import os
import matplotlib.pylab as plt 

# github sample 주소 : https://github.com/opencv/opencv/tree/master/samples/data
def get_sample(filename):
    if not os.path.exists(filename):
        url = f"https://raw.githubusercontent.com/opencv/opencv/master/samples/data/{filename}"
        urllib.request.urlretrieve(url, filename)
    return cv.imread(filename)

# 파일 읽기
img = get_sample("orange.jpg")
img_gray = cv.imread("orange.jpg", cv.IMREAD_GRAYSCALE) #  BGR 
#img_gray = get_sample("starry_night.jpg", cv.IMREAD_GRAYSCALE)
if img is None:
    sys.exit("Could not read the image.")

#cv.imshow("Display window_color", img)
#cv.imshow("Display window_gray", img_gray)

# 히스토그램 계산
hist = cv.calcHist([img_gray], [0], None, [256], [0, 256])

# 히스토그램 그리기
plt.plot(hist)
print("hist.shape:", hist.shape)  # (256, 1)
print("hist.sum():", hist.sum(), "img.shape:", img_gray.shape)
plt.show()

# 창 닫기 
k = cv.waitKey(0)

# 파일 저장 
if k == ord("s"):
    cv.imwrite("orange.png", img)