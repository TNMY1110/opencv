import cv2 as cv
import numpy as np

# 웹캠 연결
cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("웹캠을 열 수 없습니다")
    exit()

cv.namedWindow('Line Tracing Stage 2', cv.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    _, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    # --- 새로 추가: 노이즈 제거 ---
    # 메디안 필터
    binary = cv.medianBlur(binary, 5)
    
    # 모폴로지 열기 (작은 노이즈 제거)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    binary = cv.morphologyEx(binary, cv.MORPH_OPEN, kernel)

    contours, _ = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    largest_cnt = None
    max_area = 0
    for cnt in contours:
        area = cv.contourArea(cnt)
        if area > max_area:
            max_area = area
            largest_cnt = cnt

    if largest_cnt is not None and max_area > 100:
        M = cv.moments(largest_cnt)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            # 컨투어 + 중심점 그리기
            cv.drawContours(frame, [largest_cnt], 0, (0, 255, 0), 2)
            cv.circle(frame, (cx, cy), 8, (0, 0, 255), -1)

            # --- 새로 추가: 방향 계산 ---
            # fitLine으로 최적선 계산
            vx, vy, x, y = cv.fitLine(largest_cnt, cv.DIST_L2, 0, 0.01, 0.01)
            
            # 각도 계산
            angle = np.arctan2(vy, vx) * 180 / np.pi

            # --- 새로 추가: 제어신호 생성 ---
            frame_center_x = frame.shape[1] // 2
            error = cx - frame_center_x
            steer = error / frame_center_x  # -1.0 ~ 1.0

            # 정보 표시
            cv.putText(frame, f'Center: ({cx}, {cy})', (10, 30),
                      cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv.putText(frame, f'Angle: {angle:.1f} deg', (10, 60),
                      cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv.putText(frame, f'Steer: {steer:.2f}', (10, 90),
                      cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # 결과 표시
    binary_color = cv.cvtColor(binary, cv.COLOR_GRAY2BGR)
    result = np.hstack([binary_color, frame])
    cv.imshow('Line Tracing Stage 2', result)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()