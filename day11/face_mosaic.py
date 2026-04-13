import cv2
import numpy as np
import os

# Cascade 분류기 로드 함수
def load_cascade(cascade_name='haarcascade_frontalface_default.xml'):
    """Haar Cascade 분류기 로드"""
    cascade_path = cv2.data.haarcascades + cascade_name
    cascade = cv2.CascadeClassifier(cascade_path)

    if cascade.empty():
        print(f"Error: {cascade_name} 로드 실패")
        return None

    return cascade

# 폴더 생성 함수
def create_folders(paths):
    """필요한 폴더 생성"""
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)

def apply_mosaic(frame, faces, block_size=15):
    """
    얼굴 영역에 모자이크 처리
    
    Args:
        frame: 원본 이미지
        faces: Cascade로 검출된 얼굴 목록
        block_size: 모자이크 블록 크기
    
    Returns:
        모자이크 처리된 이미지
    """
    for (x, y, w, h) in faces:
        # 얼굴 영역 추출
        face_roi = frame[y:y+h, x:x+w]
        
        # 축소 (다운샘플링)
        small = cv2.resize(face_roi, (block_size, block_size)) # block_size ^ 2의 크기만큼 줄이는 식이기에 블럭 사이즈가 클수록 덜 뭉개짐
        
        # 다시 확대 (업샘플링) - INTER_NEAREST: 블록화 효과
        mosaic = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
        
        # 모자이크된 영역을 원본에 복사
        frame[y:y+h, x:x+w] = mosaic
    
    return frame
# Cascade 로드
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# 웹캠 시작
cap = cv2.VideoCapture(0)
print("웹캠 모자이크 처리 시작... (q로 종료)")

while True:
    ret, frame = cap.read()
    
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    
    # 모자이크 적용
    frame = apply_mosaic(frame, faces, block_size=15)
    
    cv2.imshow('Face Mosaic', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()