import cv2 as cv
import numpy as np

# 웹캠 연결
cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("웹캠을 열 수 없습니다")
    exit()

# 창 생성
cv.namedWindow('Line Tracing Stage 1', cv.WINDOW_AUTOSIZE)

# 반복문: 웹캠 프레임 처리
while True:
    # 프레임 읽기
    ret, frame = cap.read()
    if not ret:
        break

    # 그레이스케일 변환
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    # gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    lower_black = np.array([0, 0, 0])
    upper_black = np.array([179, 70, 70])  # S, V 둘 다 낮아야 검정

    binary = cv.inRange(hsv, lower_black, upper_black)
    # 이진화 (Otsu 사용)
    # _, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    # adapt = cv.adaptiveThreshold(binary, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)

    # 컨투어 검출
    contours, _ = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # 가장 큰 컨투어 찾기
    largest_cnt = None
    max_area = 0
    for cnt in contours:
        area = cv.contourArea(cnt)
        if area > max_area:
            max_area = area
            largest_cnt = cnt

    # 중심좌표 계산 (최소 크기 필터)
    if largest_cnt is not None and max_area > 500:
        M = cv.moments(largest_cnt)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            
            # 컨투어 그리기
            cv.drawContours(frame, [largest_cnt], 0, (0, 255, 0), 2)
            
            # 중심점 그리기 (빨강)
            cv.circle(frame, (cx, cy), 8, (0, 0, 255), -1)
            
            # 좌표 표시
            cv.putText(frame, f'Center: ({cx}, {cy})', (10, 30),
                      cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # 이진화 이미지 + 원본 나란히 표시
    binary_color = cv.cvtColor(binary, cv.COLOR_GRAY2BGR)
    result = np.hstack([binary_color, frame])
    cv.imshow('Line Tracing Stage 1', result)

    # 'q' 키로 종료
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# 정리
cap.release()
cv.destroyAllWindows()
