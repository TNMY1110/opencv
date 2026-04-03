# 라이브러리 import
import cv2 as cv
import numpy as np
import serial
import time

# 아두이노 시리얼 연결 (COM 포트, 9600 속도)
ser = serial.Serial('COM3', 9600, timeout=1)

# 웹캠을 열기
cap = cv.VideoCapture(0)

# 색상 범위 설정 (과제 1에서 확인한 값)
lower_color = np.array([50, 0, 0])
upper_color = np.array([100, 255, 255])

lower_red = np.array([0, 0, 0])
upper_red = np.array([30, 255, 255])

# 감지 면적 임계값 설정
MIN_AREA = 5000

cur_state = False
before_state = False

# 변수 초기화
prev_time = time.time()     # 이전 시간
fps = 0                     # fps
fps_list = []               # fps 리스트

reaction_times = []         # 반응 시간 기록 리스트
state_change_time = None    # 상태 변화 감지 시점

# 함수
def send_command(ser, command):
    try:
        ser.write((command + '\n').encode('utf-8'))
        return True  # 블로킹 없음
    
    except Exception as e: # 오류시
        return False

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

    cur_time = time.time()
    elapsed = cur_time - prev_time

    # 0 나누기 방지
    if elapsed > 0:
        fps_list.append(1 / elapsed)

    if len(fps_list) > 30:      # 최근 30프레임 평균
        fps_list.pop(0)         # 넘어가면 제거

    fps = sum(fps_list) / len(fps_list)     # 평균 구하기
    prev_time = cur_time

#   HSV 색공간으로 변환
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

#   마스크 생성
    mask_green = cv.inRange(hsv, lower_color, upper_color)
    mask_red = cv.inRange(hsv, lower_red, upper_red)

    mask = cv.bitwise_or(mask_green, mask_red)

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
        cur_state = True
        cv.putText(res, f"Area: {area:.0f}", (10, 30),
                cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    else:
        cur_state = False
        cv.putText(res, f"Area: {area:.0f}", (10, 30),
                cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # fps 표시
    cv.putText(frame, f"FPS: {fps:.1f}", (10, 30),
               cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    cv.imshow('frame',frame)
    cv.imshow('mask',mask)
    cv.imshow('res',res)

#   상태가 이전 상태와 다르면 아두이노에 명령 전송
    if cur_state != before_state:
        state_change_time = time.time()

        if cur_state:
            com_result = send_command(ser, 'O')
            
        else:
            com_result = send_command(ser, 'C')

        if com_result:
            reaction_time = (time.time() - state_change_time) * 1000  # ms 단위
            reaction_times.append(reaction_time)

            before_state = cur_state

            print(f"✅ PASS: 반응 시간: {reaction_time:.2f}ms")
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

# 개선 1: FPS 표시로 시스템 성능 모니터링
# 개선 2: 반응 시간 기록으로 성능 분석
# 개선 3: 여러 색상을 동시에 감지