# 라이브러리 import
import cv2 as cv
import numpy as np

# 웹캠을 열기
cap = cv.VideoCapture(0)

# 감지할 색상의 HSV 범위 설정
lower_color = np.array([50, 0, 0])
upper_color = np.array([100, 255, 255])

# 감지 면적 임계값 설정
MIN_AREA = 5000

def detect_color(frame):
    # 마스크 픽셀 면적 계산
    contours, _ = cv.findContours(frame, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # 면적과 임계값 비교하여 상태 결정
    for contour in contours:
        area = cv.contourArea(contour)
        if area >= MIN_AREA:
            return True, area
        
    return False, 0

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
    
    # 마스크 생성 (특정 색상만 추출)
    mask = cv.inRange(hsv, lower_color, upper_color)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)   # 작은 노이즈 제거
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)  # 구멍 메우기
    res = cv.bitwise_and(frame,frame, mask= mask)

    # 마스크 픽셀 면적 계산
    result, area = detect_color(mask)

    if area > 0:
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