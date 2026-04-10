"""
[Step 2] 이미지 전처리 추가
- Step 1에서 만든 기본 코드에 enhance_frame() 함수 추가
- 어두운 환경에서도 QR을 잘 감지하도록 개선

목표:
  1. enhance_frame() 함수 구현 (그레이스케일 + 명도 조정 + CLAHE)
  2. pyzbar.decode(detect_frame) 사용 (원본 대신 전처리된 이미지)
  3. 전처리 여부를 설정으로 선택 가능하게

개선 원리:
  - 그레이스케일: 컬러 정보 제거 (바코드는 흑백만 중요)
  - 명도 조정: 어두운 부분을 밝게
  - CLAHE: 로컬 대비(contrast) 강화 (인접한 픽셀끼리만 비교)
"""

import cv2
from pyzbar import pyzbar

# ==========================================
# 설정
# ==========================================

ENHANCE_IMAGE = False  # True면 전처리 ON, False면 OFF

# ==========================================
# 1️⃣ 이미지 전처리 함수
# ==========================================

def enhance_frame(frame):
    """
    바코드 감지 향상을 위한 이미지 전처리

    입력:
      frame: 원본 컬러 이미지 (BGR)

    출력:
      enhanced: 전처리된 그레이스케일 이미지
    """

    # TODO: 단계 1 - 그레이스케일로 변환
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # TODO: 단계 2 - 명도 조정 (어두운 부분을 밝게)
    brightened = cv2.convertScaleAbs(gray, alpha=1.3, beta=20)
    #       - alpha: 명도 배율 (1.3 = 30% 더 밝게)
    #       - beta: 절대 밝기 추가 (20 = 추가로 +20)

    # TODO: 단계 3 - CLAHE로 대비 강화
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(brightened)

    return enhanced  # 전처리된 이미지 반환


# ==========================================
# 2️⃣ 웹캠 열기
# ==========================================

cap = cv2.VideoCapture(2)

if not cap.isOpened():
    print("[경고] 웹캠 2번 실패, 0번으로 시도...")
    cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("[오류] 웹캠을 열 수 없습니다.")
    exit()

print("QR Code Scanner 시작 - Step 2: 전처리")
print(f"  ENHANCE_IMAGE = {ENHANCE_IMAGE}")
print("  q: 종료")

# ==========================================
# 3️⃣ 메인 루프
# ==========================================

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # ==========================================
    # 4️⃣ 이미지 전처리 선택
    # ==========================================

    # TODO: ENHANCE_IMAGE 값에 따라 다른 이미지 사용
    #       - True: enhance_frame(frame)으로 전처리된 이미지 사용
    #       - False: frame 원본 그대로 사용
    if ENHANCE_IMAGE:
        detect_frame = enhance_frame(frame)
    
    else:
        detect_frame = frame

    # ==========================================
    # 5️⃣ QR 코드 감지
    # ==========================================

    # TODO: pyzbar.decode(detect_frame)으로 감지
    #       (detect_frame은 전처리된 이미지 또는 원본)
    qr_codes = pyzbar.decode(detect_frame)
    
    # ==========================================
    # 6️⃣ 감지된 각 QR 처리
    # ==========================================

    for qr in qr_codes:
        # TODO: qr.type이 'QRCODE'인지 확인
        if qr.type != 'QRCODE':
            continue

        # TODO: qr.rect에서 (x, y, w, h) 추출하고 테두리 그리기
        (x, y, w, h) = qr.rect
        cv2.rectangle(detect_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # TODO: 내용 추출 및 출력
        qr_data = qr.data.decode('utf-8')
        cv2.putText(detect_frame, qr_data, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

        # TODO: 콘솔에 [감지 QR] {내용} 출력
        print(f"[감지 QR] {qr_data}")
        pass

    # ==========================================
    # 7️⃣ 화면 표시
    # ==========================================

    # TODO: cv2.imshow('QR Code Scanner', frame)
    cv2.imshow('QR Code Scanner', detect_frame)

    # TODO: 키 입력 처리 (q 키로 종료)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ==========================================
# 8️⃣ 정리
# ==========================================

cap.release()
cv2.destroyAllWindows()

print("\n[완료] Step 2 종료")
print("\n📊 비교 관찰:")
print("  1. ENHANCE_IMAGE = True → 어두운 환경에서도 감지")
print("  2. ENHANCE_IMAGE = False → 밝은 환경에서만 감지")
print("  💡 전처리 과정의 중요성을 확인했습니다!")