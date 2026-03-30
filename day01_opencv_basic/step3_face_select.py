import numpy as np
import cv2 as cv

drawing = False # true if mouse is pressed
ix,iy = -1,-1

img = cv.imread("./save/my_id_card.png")
img_copy = img.copy()  # 백 버퍼 역할 (원본 보존)

# mouse callback function
def draw_circle(event,x,y,flags,param):
    global ix, iy, drawing, img, img_copy, cropped

    if event == cv.EVENT_LBUTTONDOWN:
        drawing = True
        ix,iy = x,y

    elif event == cv.EVENT_MOUSEMOVE:
        if drawing == True:
            img = img_copy.copy()  # 매 프레임 원본으로 초기화
            cv.rectangle(img, (ix,iy), (x,y), (0,255,0), 2)

    elif event == cv.EVENT_LBUTTONUP:
        drawing = False
        cv.rectangle(img, (ix,iy), (x,y), (0,255,0), 2)

        x1, y1 = min(ix, x), min(iy, y)  # 드래그 방향 상관없이 정렬
        x2, y2 = max(ix, x), max(iy, y)

        font = cv.FONT_HERSHEY_SIMPLEX
        text = 'face'

        # getTextSize의 반환값은 (너비, 높이), baseline이므로 (text_w, text_h), _에 받음
        (text_w, text_h), _ = cv.getTextSize(text, font, 1, 1)
        text_x = (x1 + x2) // 2 - text_w // 2  # 가로 중앙
        text_y = y1 + text_h + 5               # 상단에서 약간 아래
        
        cv.putText(img, text, (text_x, text_y), font, 1, (255,255,255), 1, cv.LINE_AA)

        cropped = img[y1:y2, x1:x2]

        img_copy = img.copy()

cv.namedWindow('image')
cv.setMouseCallback('image',draw_circle)
 
while(1):
    cv.imshow('image',img)
    k = cv.waitKey(1) & 0xFF

    if k == ord("s"):
        cv.imwrite("./save/my_id_card_final.png", img)
        cv.imwrite("./save/my_id_card_final_sq.png", cropped)
        break
    elif k == ord("q"):
        break

cv.destroyAllWindows()