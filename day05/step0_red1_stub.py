import cv2 as cv
import numpy as np

# --- stub 함수: 아직 구현하지 않음 ---
def detect_color(frame):
    """특정 색상 감지 (GREEN에서 구현할 예정)"""
    return False   # 지금은 항상 False 반환


# --- 기대값 확인 ---
cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("❌ 웹캠을 열 수 없습니다")
    exit()

ret, frame = cap.read()
if not ret:
    print("❌ 프레임을 읽을 수 없습니다")
    cap.release()
    exit()

# detect_color() 함수가 아직 구현되지 않았으므로 False 반환
result = detect_color(frame)

# RED 단계: 이 부분이 실행되어야 함 (즉, FAIL이 출력되어야 함)
if result:
    print("✅ PASS: 색상 감지 성공!")
else:
    print("❌ FAIL: detect_color() 함수가 아직 구현되지 않았습니다")

cap.release()