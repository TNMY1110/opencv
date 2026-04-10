"""
[Step 3] 고급 기능 추가
- Step 2에서 만든 코드에 실무용 기능 추가
- 중복 감지 방지, 로그 파일 기록, URL 자동 열기

목표:
  1. detected_set으로 중복 감지 방지 (같은 QR은 한 번만 처리)
  2. SAVE_LOG=True면 scan_log.txt 파일에 기록
  3. AUTO_OPEN_URL=True면 감지한 URL을 자동으로 브라우저 열기
  4. 키 조작 확장 (s: 저장, c: 초기화)
  5. 화면에 감지 누계 표시

비즈니스 활용:
  - 편의점 POS: 같은 상품을 여러 번 스캔해도 한 번만 계산
  - 매출 통계: scan_log.txt에 기록된 데이터로 분석
  - 스마트폰 결제: QR 코드의 URL을 자동으로 결제 페이지로 열기
"""

import cv2
import webbrowser
from pyzbar import pyzbar
import time

# ==========================================
# 설정
# ==========================================

AUTO_OPEN_URL = True  # True면 URL 감지 시 자동으로 브라우저 열기
SAVE_LOG = True        # True면 감지된 내용을 scan_log.txt에 저장
ENHANCE_IMAGE = True   # True면 이미지 전처리로 감지율 향상

# 색상 정의
COLOR_QR = (0, 255, 0)       # QR 코드 — 초록
COLOR_TEXT = (255, 255, 255) # 텍스트 — 흰색

# ==========================================
# 1️⃣ 이미지 전처리 함수
# ==========================================

def enhance_frame(frame):
    """바코드 감지 향상을 위한 이미지 전처리"""
    # TODO: 그레이스케일 변환
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # TODO: 명도 조정
    alpha = 1.3  # 명도 증가
    beta = 20    # 밝기 추가
    enhanced = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)

    # TODO: CLAHE로 대비 강화
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(enhanced)

    return enhanced

# ==========================================
# 2️⃣ QR 코드 그리기 함수
# ==========================================

def draw_qr(frame, qr_code, color):
    """감지된 QR 코드에 테두리와 텍스트를 그린다"""
    # TODO: 폴리곤 테두리 (QR이 기울어진 경우도 정확히 그림)
    pts = qr_code.polygon
    if len(pts) == 4:
        pts_array = [(p.x, p.y) for p in pts]
        for i in range(4):
            cv2.line(frame, pts_array[i], pts_array[(i+1) % 4], color, 2)

    # TODO: 사각형 테두리 (항상 표시)
    x, y, w, h = qr_code.rect
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    # TODO: QR 코드 내용 텍스트
    data = qr_code.data.decode('utf-8')
    cv2.putText(frame, data, (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_TEXT, 2)

    # TODO: 함수가 QR 내용을 반환해야 함
    return data

# ==========================================
# 3️⃣ 웹캠 열기
# ==========================================

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[경고] 웹캠 0번 실패, 2번으로 시도...")
    cap = cv2.VideoCapture(2)

if not cap.isOpened():
    print("[오류] 웹캠을 열 수 없습니다.")
    exit()

print("QR Code Scanner 시작 - Step 3: 고급 기능")
print("  q: 종료 | s: 화면 저장 | c: 이력 초기화")
print(f"  AUTO_OPEN_URL = {AUTO_OPEN_URL}")
print(f"  ENHANCE_IMAGE = {ENHANCE_IMAGE}")

# ==========================================
# 4️⃣ 상태 관리 변수
# ==========================================

# TODO: detected_set = set() — 중복 감지 방지용 집합
detected_set = set()
frame_count = 0

# TODO: SAVE_LOG가 True면 파일 열기
log_file = None
if SAVE_LOG:
    log_file = open('scan_log.txt', 'a')

# ==========================================
# 5️⃣ 메인 루프
# ==========================================

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame_count += 1

    # ==========================================
    # 6️⃣ 이미지 전처리
    # ==========================================

    # TODO: ENHANCE_IMAGE 값에 따라 다른 이미지 사용
    if ENHANCE_IMAGE:
        detect_frame = enhance_frame(frame)
    else:
        detect_frame = frame

    # ==========================================
    # 7️⃣ QR 코드 감지
    # ==========================================

    # TODO: pyzbar.decode(detect_frame)로 감지
    qr_codes = pyzbar.decode(detect_frame, symbols=[pyzbar.ZBarSymbol.QRCODE])

    # ==========================================
    # 8️⃣ 감지된 각 QR 처리
    # ==========================================

    for qr in qr_codes:
        # TODO: qr.type이 'QRCODE'인지 확인 (바코드는 제외)
        if qr.type != 'QRCODE':
            continue

        # TODO: 내용 추출
        # TODO: draw_qr()으로 그리기
        data = draw_qr(frame, qr, COLOR_QR)

        # ==========================================
        # 9️⃣ 중복 감지 방지 (새로운 QR만 처리)
        # ==========================================

        # TODO: data가 detected_set에 없으면:
        if data not in detected_set:
        #   1. detected_set에 추가
            detected_set.add(data)
        #   2. 콘솔에 [감지 QR] {data} 출력
            print(f"[감지 QR] {data}")
        #   3. SAVE_LOG=True면 파일에 기록
            if SAVE_LOG:
                log_file.write(data + "\n")

        #   4. AUTO_OPEN_URL=True이고 URL이면 브라우저 열기
            if AUTO_OPEN_URL:
                if data.startswith("http"):
                    webbrowser.open_new_tab(data)

    # ==========================================
    # 🔟 화면에 감지 누계 표시
    # ==========================================

    # TODO: cv2.putText()로 'QR 감지 누계: {len(detected_set)}건' 표시
    #       좌표 (10, 30), 글꼴 HERSHEY_SIMPLEX, 크기 0.7, 초록색(0, 255, 0)
    cv2.putText(frame, f'QR 감지 누계: {len(detected_set)}건', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # ==========================================
    # 1️⃣1️⃣ 화면 표시
    # ==========================================

    # TODO: cv2.imshow('QR Code Scanner', frame)
    cv2.imshow('QR Code Scanner', frame)
    # ==========================================
    # 1️⃣2️⃣ 키 입력 처리
    # ==========================================

    # TODO: cv2.waitKey(1) & 0xFF로 키 입력 받기
    #       - 'q': 루프 탈출 (종료)
    #       - 's': 현재 프레임 저장 (파일명: 'scan_{타임스탬프}.png')
    #       - 'c': detected_set 초기화 (감지 이력 삭제)
    key =  cv2.waitKey(1)

    if key & 0xFF == ord('q'):
        break
    
    elif key & 0xFF == ord('s'):
        cv2.imwrite(f'./img/scan_{int(time.time())}.png', frame)
        continue

    elif key & 0xFF == ord('c'):
        detected_set.clear()
        continue

# ==========================================
# 1️⃣3️⃣ 정리
# ==========================================

cap.release()
cv2.destroyAllWindows()

if SAVE_LOG:
    log_file.close()

# ==========================================
# 1️⃣4️⃣ 최종 결과 출력
# ==========================================

# TODO: print(f"\n총 {len(detected_set)}개 QR 코드 감지:")
# TODO: for item in detected_set 루프로 각 항목 출력

print("\n[완료] Step 3 종료")
print("\n📋 실무 기능 체크리스트:")
print("  ✓ 중복 감지 방지 (같은 QR은 한 번만 처리)")
print("  ✓ 로그 파일 기록 (scan_log.txt)")
print("  ✓ URL 자동 열기 (AUTO_OPEN_URL=True)")
print("  ✓ 키 조작 확장 (s, c)")
print("  ✓ 감지 누계 표시")