# 라이브러리 import
import cv2 as cv
import numpy as np
import serial
import time

# 아두이노 시리얼 연결 (COM 포트, 9600 속도)
ser = serial.Serial('/dev/ttyUSB0', 9600, timeout=1)
time.sleep(2)  # 리셋

# 웹캠을 열기
cap = cv.VideoCapture(0)

# 색상 범위 설정 (과제 1에서 확인한 값)
lower_color = np.array([50, 0, 0])
upper_color = np.array([179, 255, 255])

# 감지 면적 임계값 설정
MIN_AREA = 5000

cur_state = 0       # 추가 설정 예정
before_state = 0    # 추가 설정 예정

# 함수
def send_command(ser, command):
    return False   # 지금은 항상 False 반환

def detect_color(frame):
    # 마스크 픽셀 면적 계산
    contours, _ = cv.findContours(frame, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    if not contours:
        return False, 0

    max_area = max(cv.contourArea(contour) for contour in contours)
        
    return max_area >= MIN_AREA, max_area

# 반복:
while(1):
#   웹캠에서 프레임 읽기
    ret, frame = cap.read()

    if not ret:
        print("❌ 프레임을 읽을 수 없습니다")
        cap.release()
        exit()

#   HSV 색공간으로 변환
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

#   마스크 생성
    mask = cv.inRange(hsv, lower_color, upper_color)

#   모폴로지 연산으로 노이즈 제거
    kernel = np.ones((5, 5), np.uint8)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)   # 작은 노이즈 제거
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)  # 구멍 메우기

    h, w = frame.shape[:2]      # 높이, 너비
    rect_w, rect_h = w // 2, h // 2   # 사각형 높이, 너비

#   좌상단 좌표
    x1 = w // 2 - rect_w // 2
    y1 = h // 2 - rect_h // 2 

#   우하단 좌표
    x2 = w // 2 + rect_w // 2
    y2 = h // 2 + rect_h // 2

#   (np.zeros((h, w), dtype=np.uint8) 전부 검정(0)으로 채워진 빈 영역 생성
    roi_mask = np.zeros((h, w), dtype=np.uint8)

#   사각형 영역만 흰색(255)으로 채움
    roi_mask[y1:y2, x1:x2] = 255
    
#   bitwise_and 함수로 일치하는 부분만 남김
    mask = cv.bitwise_and(mask, roi_mask)
    res = cv.bitwise_and(frame,frame, mask= mask)

    cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv.rectangle(res, (x1, y1), (x2, y2), (0, 255, 0), 2)

#   마스크 픽셀 면적 계산
    result, area = detect_color(mask)

#   현재 상태를 화면에 표시
#   현재는 최소 크기 이상의 색 있는 물체가 감지됐을 경우 화면에 초록색으로 픽셀 크기 출력
#   면적과 임계값 비교하여 상태 결정
    if area > MIN_AREA:
        cv.putText(res, f"Area: {area:.0f}", (10, 30),
                cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    else:
        cv.putText(res, f"Area: {area:.0f}", (10, 30),
                cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    cv.imshow('frame',frame)
    cv.imshow('mask',mask)
    cv.imshow('res',res)

#   상태가 이전 상태와 다르면 아두이노에 명령 전송
    if cur_state != before_state:
        com_result = send_command(ser, 'O')

        if com_result:
            print("✅ PASS: 아두이노 명령 전송 성공!")
        else:
            print("❌ FAIL: send_command() 함수가 아직 구현되지 않았습니다")

    # 'q' 키 입력 시 루프 종료
    k = cv.waitKey(5)
    
    if k == ord('q'):
        # 리소스 해제
        cap.release()
        break

cv.destroyAllWindows()
