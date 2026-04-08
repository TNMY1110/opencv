import cv2
import numpy as np

# HOG 디스크립터 설정
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# 최적화된 버전: 매 프레임마다 처리하지 않고, N프레임마다 처리
PROCESS_INTERVAL = 3  # 3프레임마다 검출 수행

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: 웹캠을 열 수 없습니다")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

frame_count = 0
last_detections = []
processing_times = []

import time

print("최적화된 실시간 처리 시작...")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    
    # N프레임마다 검출 수행 (속도 향상)
    if frame_count % PROCESS_INTERVAL == 0:
        start_time = time.time()
        
        detections, weights = hog.detectMultiScale(frame, winStride=(8, 8), padding=(16, 16), scale=1.05)
        
        last_detections = [
            (x, y, w, h) for (x, y, w, h), w in zip(detections, weights) if w > 0.5
        ]
        
        elapsed = time.time() - start_time
        processing_times.append(elapsed)
    
    # 바운딩 박스 표시
    for (x, y, w, h) in last_detections:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    # FPS 계산
    if len(processing_times) > 0:
        avg_time = np.mean(processing_times[-10:])  # 최근 10개 평균
        fps = 1.0 / avg_time if avg_time > 0 else 0
    else:
        fps = 0
    
    cv2.putText(frame, f"Frame: {frame_count}, FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"Detected: {len(last_detections)}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.imshow('Optimized Pedestrian Detection', frame)
    
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()

print(f"\n최적화 결과:")
print(f"처리 시간: {np.mean(processing_times)*1000:.1f}ms")
print(f"평균 FPS: {1.0/np.mean(processing_times):.1f}")