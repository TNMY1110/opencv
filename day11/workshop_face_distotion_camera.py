import cv2
import numpy as np
import dlib
import os

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

LANDMARKS = {
    'left_eye':  list(range(36, 42)),
    'right_eye': list(range(42, 48)),
    'nose':      list(range(27, 36)),
    'mouth':     list(range(48, 68)),
}

def bulge(image, cx, cy, radius, strength=0.5):
    """
    (cx, cy) 중심으로 radius 범위를 부풀림
    strength: 부풀림 강도 (양수 = 볼록, 음수 = 오목)
    """
    h, w = image.shape[:2]
    
    # 영향 받는 영역 계산 (경계 클리핑)
    x1 = max(0, cx - radius)
    y1 = max(0, cy - radius)
    x2 = min(w, cx + radius)
    y2 = min(h, cy + radius)
    
    if x2 - x1 <= 0 or y2 - y1 <= 0:
        return image
    
    # 영역 내 픽셀 좌표 그리드 생성
    ys, xs = np.mgrid[y1:y2, x1:x2]
    
    # 중심으로부터의 거리
    dx = xs - cx
    dy = ys - cy
    dist = np.sqrt(dx**2 + dy**2).astype(np.float32)
    
    # radius 밖은 변환 없음
    mask = dist < radius
    
    # 부풀림 공식: 중심에 가까울수록 더 많이 당김
    factor = np.ones_like(dist)
    factor[mask] = 1 - strength * (1 - dist[mask] / radius) ** 2
    
    # 새로운 픽셀 좌표 계산
    map_x = (cx + dx * factor).astype(np.float32)
    map_y = (cy + dy * factor).astype(np.float32)
    
    # 경계 클리핑
    map_x = np.clip(map_x, 0, w - 1)
    map_y = np.clip(map_y, 0, h - 1)
    
    # 왜곡 적용
    result = image.copy()
    region = cv2.remap(
        image, map_x, map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT
    )
    result[y1:y2, x1:x2] = region
    
    return result


def get_center_and_radius(landmarks, indices, scale=1.5):
    """특정 랜드마크 인덱스들의 중심좌표와 반경 반환"""
    pts = landmarks[indices]
    cx = int(pts[:, 0].mean())
    cy = int(pts[:, 1].mean())
    
    # 반경 = 점들의 최대 거리 * scale
    dists = np.sqrt((pts[:, 0] - cx)**2 + (pts[:, 1] - cy)**2)
    radius = int(dists.max() * scale)
    
    return cx, cy, max(radius, 10)  # 최소 반경 10


def apply_distortion(frame, gray, strength=0.5):
    faces = detector(gray)
    
    if len(faces) == 0:
        cv2.putText(frame, "No face detected", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return frame
    
    result = frame.copy()
    
    for face in faces:
        lm = predictor(gray, face)
        landmarks = np.array([[p.x, p.y] for p in lm.parts()])
        
        # 눈/코/입 각각 부풀리기
        for part, indices in LANDMARKS.items():
            cx, cy, radius = get_center_and_radius(landmarks, indices)
            result = bulge(result, cx, cy, radius, strength)
    
    return result

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

# Cascade 로드
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# 웹캠 시작
cap = cv2.VideoCapture(0)
strength = 0.5  # 초기 강도

print("웹캠 왜곡 처리 시작... (q로 종료)")
print("  +/- 키: 강도 조절")

while True:
    ret, frame = cap.read()
    
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    result = apply_distortion(frame, gray, strength=strength)
    
    # 강도 표시
    cv2.putText(result, f"Strength: {strength:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow('Face Distortion', result)
    
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('+') or key == ord('='):
        strength = min(2.0, strength + 0.1)  # 최대 1.0
    elif key == ord('-'):
        strength = max(-1.0, strength - 0.1)  # 최소 0.1

cap.release()
cv2.destroyAllWindows()