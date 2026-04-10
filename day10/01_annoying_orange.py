import cv2
import dlib
import numpy as np
from imutils import face_utils 

# 오렌지 이미지 로드 
orange_img = cv2.imread('./img/orange.jpg')
orange_img = cv2.resize(orange_img, (512, 512))

# dlib : 얼굴 감지기 초기화 
detector = dlib.get_frontal_face_detector()

# dlib : 랜드마크 예측기 초기화 (모델 파일 )
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# 웹캠 시작 
#cap = cv2.VideoCapture(0) 
cap = cv2.VideoCapture(0) 

def get_roi(img, points, margin_ratio=0.18):
    x1 = points[:, 0].min()
    y1 = points[:, 1].min()
    x2 = points[:, 0].max()
    y2 = points[:, 1].max()

    margin = int((x2 - x1) * margin_ratio)

    # 음수 인덱스 방지
    x1 = max(0, x1 - margin)
    y1 = max(0, y1 - margin)
    x2 = min(img.shape[1], x2 + margin)
    y2 = min(img.shape[0], y2 + margin)

    return img[y1:y2, x1:x2].copy(), ((x1+x2)//2, (y1+y2)//2)

def clone_to_orange(src, dst_img, dst_center):
    if src.size == 0:
        return dst_img
    try:
        mask = np.full(src.shape[:2], 255, dtype=np.uint8)
        return cv2.seamlessClone(src, dst_img, mask, dst_center, cv2.MIXED_CLONE)
    except Exception as e:
        print(f"seamlessClone 실패: {e}")
        return dst_img

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (512, 512))

    # 그레이 스케일 변환 (얼굴 감지는 색상 불필요)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 얼굴 감지
    faces = detector(gray, 0)

    if len(faces) == 0:
        #얼굴이 없으면 원본 프레임 출력 
        cv2.imshow('result', frame)
        continue

    result = orange_img.copy()
    
    for face in faces:
        # 얼굴 영역에서 68개의 랜드마크 좌표 예측 
        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape) # Numpy 배열로 변환  

    # 눈 영역 
    left_eye = shape[36:42] # 왼쪽 눈 
    right_eye = shape[42:48] # 오른쪽 눈 
    mouth = shape[48:58] # 입  

    left_eye_img,  le_center = get_roi(frame, left_eye)
    right_eye_img, re_center = get_roi(frame, right_eye)
    mouth_img,     mo_center = get_roi(frame, mouth, margin_ratio=0.1)

    # 오렌지에 붙여넣을 고정 좌표 (오렌지 이미지 기준)
    le_dst = (130, 190)
    re_dst = (300, 190)
    mo_dst = (210, 340)

    result = clone_to_orange(left_eye_img,  result, le_dst)
    result = clone_to_orange(right_eye_img, result, re_dst)
    result = clone_to_orange(mouth_img,     result, mo_dst)

    cv2.imshow('result', result)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()