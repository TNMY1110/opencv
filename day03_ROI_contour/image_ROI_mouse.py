import cv2 as cv 
import urllib.request
import sys 
import os

# github sample 주소 : https://github.com/opencv/opencv/tree/master/samples/data
def get_sample(filename):
    if not os.path.exists(filename):
        url = f"https://raw.githubusercontent.com/opencv/opencv/master/samples/data/{filename}"
        urllib.request.urlretrieve(url, filename)
    return cv.imread(filename)

# 전역 변수 설정
isDragging = False
x0, y0, w, h = -1, -1, -1, -1
blue, red = (255,0,0), (0,0,255)

def onMouse(event, x, y, flags, param):
    global isDragging, x0, y0, img
    
    if event == cv.EVENT_LBUTTONDOWN:  # 마우스 왼쪽 버튼 다운
        isDragging = True
        x0, y0 = x, y
        
    elif event == cv.EVENT_MOUSEMOVE:  # 마우스 움직임
        if isDragging:
            img_draw = img.copy()
            cv.rectangle(img_draw, (x0, y0), (x, y), blue, 2)
            cv.imshow('img', img_draw)
            
    elif event == cv.EVENT_LBUTTONUP:  # 마우스 왼쪽 버튼 업
        if isDragging:
            isDragging = False
            w, h = x - x0, y - y0
            print(f"x:{x0}, y:{y0}, w:{w}, h:{h}")
            
            if w > 0 and h > 0:  # 올바른 드래그 방향
                img_draw = img.copy()
                cv.rectangle(img_draw, (x0, y0), (x, y), red, 2)
                cv.imshow('img', img_draw)
                
                # ROI 지정 및 표시
                roi = img[y0:y0+h, x0:x0+w]
                cv.imshow('cropped', roi)
                cv.moveWindow('cropped', 0, 0)
                cv.imwrite('./cropped.jpg', roi)
                print("cropped.")
            else:
                cv.imshow('img', img)
                print("좌측 상단에서 우측 하단으로 드래그하세요.")

# 파일 읽기
img = get_sample("orange.jpg")

# 파일 이미지 창 열기 
#cv.imshow("Display window_color", img)
cv.imshow("ROI", img)

# 마우스 이벤트 등록 
cv.setMouseCallback('ROI', onMouse)
# 창 닫기 
k = cv.waitKey(0)

# 파일 저장 
if k == ord("s"):
    cv.imwrite("orange.png", img)