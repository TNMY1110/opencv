import cv2
import numpy as np
import dlib
import os


# dlib : 얼굴 감지기 초기화 
detector = dlib.get_frontal_face_detector()

# dlib : 랜드마크 예측기 초기화 (모델 파일 )
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

def get_landmarks(gray, face_rect):
    """dlib으로 68개 랜드마크 좌표 반환"""
    shape = predictor(gray, face_rect)
    return np.array([[p.x, p.y] for p in shape.parts()])

def get_triangles(points, size):
    h, w = size
    subdiv = cv2.Subdiv2D((0, 0, w, h))
    
    for p in points:
        x = max(0, min(float(p[0]), w - 1))
        y = max(0, min(float(p[1]), h - 1))
        subdiv.insert((x, y))
    
    triangles = subdiv.getTriangleList().astype(np.int32)
    
    triangle_indices = []
    for t in triangles:
        pts = [(t[0], t[1]), (t[2], t[3]), (t[4], t[5])]
        idx = []
        valid = True
        for pt in pts:
            found = False
            for i, lm in enumerate(points):
                if abs(lm[0] - pt[0]) < 1 and abs(lm[1] - pt[1]) < 1:
                    idx.append(i)
                    found = True
                    break
            if not found:
                valid = False
                break
        if valid and len(idx) == 3:
            triangle_indices.append(idx)
    
    return triangle_indices

def warp_triangle(src, dst, t_src, t_dst):
    r_src = cv2.boundingRect(np.float32([t_src]))
    r_dst = cv2.boundingRect(np.float32([t_dst]))

    if r_src[2] == 0 or r_src[3] == 0 or r_dst[2] == 0 or r_dst[3] == 0:
        return

    t_src_rel = [(p[0] - r_src[0], p[1] - r_src[1]) for p in t_src]
    t_dst_rel = [(p[0] - r_dst[0], p[1] - r_dst[1]) for p in t_dst]

    mask = np.zeros((r_dst[3], r_dst[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(t_dst_rel), (1, 1, 1))

    M = cv2.getAffineTransform(np.float32(t_src_rel), np.float32(t_dst_rel))

    src_crop = src[r_src[1]:r_src[1]+r_src[3], r_src[0]:r_src[0]+r_src[2]]
    
    if src_crop.size == 0 or src_crop.shape[0] == 0 or src_crop.shape[1] == 0:
        return

    warped = cv2.warpAffine(src_crop, M, (r_dst[2], r_dst[3]))

    dst_region = dst[r_dst[1]:r_dst[1]+r_dst[3], r_dst[0]:r_dst[0]+r_dst[2]]

    if dst_region.shape != warped.shape or dst_region.shape != mask.shape:
        return

    dst[r_dst[1]:r_dst[1]+r_dst[3], r_dst[0]:r_dst[0]+r_dst[2]] = \
        dst_region * (1 - mask) + warped * mask
    
def apply_swap(frame, gray):
    """
    프레임에서 두 얼굴을 감지하여 스왑
    얼굴이 2개 미만이면 원본 반환
    """
    faces = detector(gray)
    
    if len(faces) < 2:
        cv2.putText(frame, "Need 2 faces!", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return frame

    # 두 얼굴의 랜드마크 추출
    lm1 = get_landmarks(gray, faces[0])
    lm2 = get_landmarks(gray, faces[1])

    h, w = frame.shape[:2]
    result = frame.copy()

    # 삼각분할 (얼굴1 기준)
    triangles = get_triangles(lm1, (h, w))

    # 얼굴1 → 얼굴2 위치로, 얼굴2 → 얼굴1 위치로
    for idx in triangles:
        t1 = [lm1[i].tolist() for i in idx]
        t2 = [lm2[i].tolist() for i in idx]
        warp_triangle(frame, result, t1, t2)  # 얼굴1 → 얼굴2 자리
        warp_triangle(frame, result, t2, t1)  # 얼굴2 → 얼굴1 자리

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

# 웹캠 시작
cap = cv2.VideoCapture(0)
print("웹캠 모자이크 처리 시작... (q로 종료)")

while True:
    ret, frame = cap.read()
    
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    result = apply_swap(frame, gray)
    
    cv2.imshow('Face Swap', result)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()