# 라이브러리 import
import cv2 as cv
import numpy as np

# 웹캠을 열기
cap = cv.VideoCapture(0)

# 감지 면적 임계값 설정
MIN_AREA = 5000

def nothing(x):
    pass

cv.namedWindow('HSV')

cv.createTrackbar('Lower_H', 'HSV', 50, 179, nothing)
cv.createTrackbar('Lower_S', 'HSV', 0, 255, nothing)
cv.createTrackbar('Lower_V', 'HSV', 0, 255, nothing)

cv.createTrackbar('Upper_H', 'HSV', 100, 179, nothing)
cv.createTrackbar('Upper_S', 'HSV', 255, 255, nothing)
cv.createTrackbar('Upper_V', 'HSV', 255, 255, nothing)

def detect_color(frame):
    # 마스크 픽셀 면적 계산
    contours, _ = cv.findContours(frame, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    if not contours:
        return False, 0

    max_area = max(cv.contourArea(contour) for contour in contours)
        
    return max_area >= MIN_AREA, max_area

if not cap.isOpened():
    print("❌ 웹캠을 열 수 없습니다")
    exit()

# 반복:
while(1):
    #   웹캠에서 프레임 읽기
    ret, frame = cap.read()

    if not ret:
        print("❌ 프레임을 읽을 수 없습니다")
        cap.release()
        exit()

    # HSV 색공간으로 변환
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    
    # 트랙바로 값 조정
    lh = cv.getTrackbarPos('Lower_H', 'HSV')
    uh = cv.getTrackbarPos('Upper_H', 'HSV')

    ls = cv.getTrackbarPos('Lower_S', 'HSV')
    us = cv.getTrackbarPos('Upper_S', 'HSV')

    lv = cv.getTrackbarPos('Lower_V', 'HSV')
    uv = cv.getTrackbarPos('Upper_V', 'HSV')

    # 감지할 색상의 HSV 범위 설정
    lower_color = np.array([lh, ls, lv])
    upper_color = np.array([uh, us, uv])

    # 마스크 생성 (특정 색상만 추출)
    mask = cv.inRange(hsv, lower_color, upper_color)

    # 모폴로지 연산으로 노이즈 제거
    kernel = np.ones((5, 5), np.uint8)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)   # 작은 노이즈 제거
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)  # 구멍 메우기

    h, w = frame.shape[:2]      # 높이, 너비
    rect_w, rect_h = w // 2, h // 2   # 사각형 높이, 너비

    # 좌상단 좌표
    x1 = w // 2 - rect_w // 2
    y1 = h // 2 - rect_h // 2 

    # 우하단 좌표
    x2 = w // 2 + rect_w // 2
    y2 = h // 2 + rect_h // 2

    # (np.zeros((h, w), dtype=np.uint8) 전부 검정(0)으로 채워진 빈 영역 생성
    roi_mask = np.zeros((h, w), dtype=np.uint8)

    # 사각형 영역만 흰색(255)으로 채움
    roi_mask[y1:y2, x1:x2] = 255
    
    # bitwise_and 함수로 일치하는 부분만 남김
    mask = cv.bitwise_and(mask, roi_mask)
    res = cv.bitwise_and(frame,frame, mask= mask)

    cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv.rectangle(res, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # 마스크 픽셀 면적 계산
    result, area = detect_color(mask)

    if area > MIN_AREA:
        cv.putText(res, f"Area: {area:.0f}", (10, 30),
                cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    else:
        cv.putText(res, f"Area: {area:.0f}", (10, 30),
                cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    cv.imshow('frame',frame)
    cv.imshow('mask',mask)
    cv.imshow('res',res)
    
    # 상태를 터미널과 화면에 표시
    if result:
        print("✅ PASS: 색상 감지 성공!")
    else:
        print("❌ FAIL: 색상 감지 실패!")

    # 'q' 키 입력 시 루프 종료
    k = cv.waitKey(5)
    
    if k == ord('q'):
        # 리소스 해제
        cap.release()
        break

cv.destroyAllWindows()