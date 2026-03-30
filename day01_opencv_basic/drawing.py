import numpy as np
import cv2 as cv
 
# Create a black image
img = np.zeros((512,512,3), np.uint8)
 
# Draw a diagonal blue line with thickness of 5 px

# cv.line(img,(0,0),(511,511),(255,0,0),5)

# cv.rectangle(img,(384,0),(510,128),(0,255,0),3)

# cv.circle(img,(447,63), 63, (0,0,255), -1)

# cv.ellipse(img,(256,256),(100,50),0,0,180,255,-1)

# pts = np.array([[10,5],[20,30],[70,20],[50,10]], np.int32)
# pts = pts.reshape((-1,1,2))
# cv.polylines(img,[pts],True,(0,255,255))

font = cv.FONT_HERSHEY_SIMPLEX
cv.putText(img,'OpenCV',(10,500), font, 4,(255,255,255),2,cv.LINE_AA)

# 좌표는 좌상단이 0,0이고 y값이 커지면 아래로 감, 색깔 순서가 BGR임ㅋㅋ
# cv.line(img,(0,0),(511,511),(0,255,255),2)
# 그릴 대상의 이미지, (중심 좌표), (장축 반지름, 단축 반지름), 타원 회전 각도, 호의 시작 각도, 호의 끝 각도, (색상), 두께(-1 = 내부 채우기)
# cv.ellipse(img,(256,256), (150, 80), 0, 0, 360,(255, 255, 255),-1)

# True = 마지막 점과 첫 점을 자동으로 연결
# pts = np.array([[256, 100], [100, 400], [400, 400]], np.int32)
# cv.polylines(img, [pts], True, color=(255, 0, 0))

cv.imshow("Drawing", img)
cv.waitKey(0)
cv.destroyAllWindows()