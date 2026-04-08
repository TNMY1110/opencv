import cv2
import numpy as np
import time

class PedestrianDetectionStats:
    """보행자 감지 통계 수집"""
    
    def __init__(self):
        self.frame_count = 0
        self.detection_count = 0
        self.max_per_frame = 0
        self.detection_history = []
        self.start_time = time.time()
    
    def update(self, detections):
        """프레임 정보 업데이트"""
        self.frame_count += 1
        self.detection_count += len(detections)
        self.max_per_frame = max(self.max_per_frame, len(detections))
        self.detection_history.append(len(detections))
    
    def print_stats(self):
        """통계 출력"""
        elapsed = time.time() - self.start_time
        fps = self.frame_count / elapsed
        
        print("\n" + "="*50)
        print("📊 보행자 감지 통계")
        print("="*50)
        print(f"처리 시간: {elapsed:.1f}초")
        print(f"처리된 프레임: {self.frame_count}개")
        print(f"평균 FPS: {fps:.1f}")
        print(f"총 감지한 보행자: {self.detection_count}명")
        print(f"프레임당 평균: {self.detection_count/max(1, self.frame_count):.2f}명")
        print(f"최대 감지: {self.max_per_frame}명/프레임")
        
        # 감지되지 않은 프레임
        zero_detection = sum(1 for d in self.detection_history if d == 0)
        print(f"보행자 없는 프레임: {zero_detection}개 ({zero_detection/self.frame_count*100:.1f}%)")

# HOG 디스크립터 설정
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# 웹캠 열기
cap = cv2.VideoCapture(0)  # 0 = 기본 웹캠

if not cap.isOpened():
    print("Error: 웹캠을 열 수 없습니다")
    exit()

# 프레임 크기 설정 (처리 속도 향상)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

frame_count = 0
detected_frames = 0

print("="*50)
print("웹캠 보행자 감지 시작")
print("'q' 또는 ESC로 종료")
print("="*50)

while True:
    ret, frame = cap.read()

    if not ret:
        break
    
    frame_count += 1
    
    # ① HOG로 보행자 감지 ---
    detections, weights = hog.detectMultiScale(
        frame,
        winStride=(8, 8),
        padding=(16, 16),
        scale=1.05
    )

    # ② 신뢰도 필터링 ---
    CONFIDENCE_THRESHOLD = 0.5
    filtered_detections = [
        (int(x), int(y), int(w), int(h))
        for (x, y, w, h), weight in zip(detections, weights)
        if weight > CONFIDENCE_THRESHOLD
    ]
    
    if len(filtered_detections) > 0:
        detected_frames += 1
    
    # ③ 바운딩 박스 그리기 ---
    for (x, y, w, h) in filtered_detections:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

   
    # ④ 정보 표시 ---
    cv2.putText(frame, f"Frame: {frame_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.putText(frame, f"Detected: {len(filtered_detections)}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.putText(frame, f"Detection Rate: {detected_frames/max(1, frame_count)*100:.1f}%", (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 200, 255), 2)
  
    # ⑤ 화면 표시 ---
    cv2.imshow('Webcam - Pedestrian Detection', frame)
    
    # ⑥ 종료 조건 ---
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q') or key == 27:  # 'q' 또는 ESC
        break

cv2.destroyAllWindows()

# 통계 출력
print(f"\n{'='*50}")
print(f"처리 완료")
print(f"{'='*50}")
print(f"총 프레임: {frame_count}개")
print(f"보행자 감지된 프레임: {detected_frames}개")
print(f"감지율: {detected_frames/max(1, frame_count)*100:.1f}%")

THRESHOLDS = [0.3, 0.5, 0.7, 0.9]

for threshold in THRESHOLDS:
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    detected_count = 0
    frame_count = 0
    
    print(f"\n신뢰도 임계값: {threshold}")
    print("-" * 40)
    
    while frame_count < 100:  # 100프레임만 처리
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        detections, weights = hog.detectMultiScale(frame, winStride=(8, 8), padding=(16, 16), scale=1.05)
        
        # 필터링
        filtered = sum(1 for w in weights if w > threshold)
        detected_count += filtered
        
        # 화면에 표시
        for (x, y, w, h), weight in zip(detections, weights):
            if weight > threshold:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        cv2.putText(frame, f"Threshold: {threshold}, Count: {filtered}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv2.imshow(f'Threshold: {threshold}', frame)
        if cv2.waitKey(1) == 27:
            break
    
    avg_detected = detected_count / max(1, frame_count)
    print(f"프레임당 평균 감지: {avg_detected:.2f}명")
    
    cap.release()
    cv2.destroyAllWindows()

print("\n💡 분석: 어떤 임계값이 가장 좋은가?")

cv2.destroyAllWindows()

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


# 사용
stats = PedestrianDetectionStats()
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    detections, weights = hog.detectMultiScale(frame, winStride=(8, 8), padding=(16, 16), scale=1.05)
    filtered = [(x, y, w, h) for (x, y, w, h), w in zip(detections, weights) if w > 0.5]
    
    stats.update(filtered)
    
    for (x, y, w, h) in filtered:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    cv2.imshow('Pedestrian Detection', frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
stats.print_stats()
