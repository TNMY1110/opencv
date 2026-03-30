import numpy as np
import cv2 as cv
import urllib.request
import sys
import os

img = cv.imread("./captures/my_photo_0.png")

if img is None:
    sys.exit("Could not read the image.")

overlay = img.copy()
# 높이 480 너비 640
cv.rectangle(overlay, (0,400), (640, 480), (0,0,0), -1)
font = cv.FONT_HERSHEY_SIMPLEX
cv.putText(img, 'Tony Stark', (10, 440), font, 1, (255,255,255), 1, cv.LINE_AA)
cv.putText(img, 'Avengers', (10, 460), font, 0.5, (255,255,255), 1, cv.LINE_AA)
cv.addWeighted(overlay, 0.5, img, 0.5, 0, img)

cv.imshow("Drawing", img)

# 이미지 정보 확인
print("=== 컬러 이미지 ===")
print(f"shape: {img.shape}") # (높이, 너비, 채널)

# 창 닫기
k = cv.waitKey(0)

# 파일 저장
if k == ord("s"):
    cv.imwrite("./save/my_id_card.png", img)
    cv.destroyAllWindows()

if k == ord('q'):
        cv.destroyAllWindows()
