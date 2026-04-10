"""
[Step 1] 기본 QR 코드 감지
- 웹캠에서 QR 코드를 읽고 화면에 표시

목표:
  1. 웹캠 열기
  2. pyzbar.decode()로 QR 감지
  3. 감지된 QR에 테두리 그리기
  4. 내용 출력하기
"""

import cv2
from pyzbar import pyzbar

# ==========================================
# 1️⃣ 웹캠 열기
# ==========================================

# TODO: cv2.VideoCapture()를 사용해서 웹캠 열기
# 조건: 웹캠이 없으면 0번 시도 → 실패하면 에러 메시지 출력
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("[오류] 웹캠을 열 수 없습니다.")
    exit()

print("QR Code Scanner 시작 - Step 1: 기본")
print("  q: 종료")

# ==========================================
# 2️⃣ 메인 루프
# ==========================================

while True:
    # TODO: cap.read()로 프레임 읽기
    ret, frame = cap.read()

    if not ret:
        continue

    # ==========================================
    # 3️⃣ QR 코드 감지
    # ==========================================
    # TODO: pyzbar.decode(frame)으로 QR 감지
    qr_codes = pyzbar.decode(frame)
    
    # ==========================================
    # 4️⃣ 감지된 각 QR에 테두리 그리기
    # ==========================================
    for qr in qr_codes:
        # TODO: qr.type이 'QRCODE'인지 확인하고, 맞으면 계속 진행
        #       (바코드는 제외)
        if qr.type != 'QRCODE':
            continue

        # TODO: qr.rect에서 (x, y, w, h) 추출
        (x, y, w, h) = qr.rect
        # TODO: cv2.rectangle()으로 녹색(0, 255, 0) 테두리 그리기
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # TODO: qr.data.decode('utf-8')로 내용 추출
        qr_data = qr.data.decode('utf-8')
        # TODO: cv2.putText()로 내용 출력 (좌표: (x, y-10))
        cv2.putText(frame, qr_data, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        # TODO: 콘솔에 [감지 QR] {내용} 형태로 출력
        print(f"[감지 QR] {qr_data}")
        pass

    # ==========================================
    # 5️⃣ 화면 표시
    # ==========================================

    # TODO: cv2.imshow('QR Code Scanner', frame)으로 화면 표시
    cv2.imshow('QR Code Scanner', frame)

    # TODO: cv2.waitKey(1) & 0xFF로 키 입력 대기
    # 조건: 'q' 키 입력 시 루프 탈출
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ==========================================
# 6️⃣ 정리
# ==========================================
# TODO: 웹캠 릴리스 및 윈도우 닫기
cap.release()
cv2.destroyAllWindows()

print("\n[완료] Step 1 종료")