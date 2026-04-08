import cv2
import numpy as np
import urllib.request
from sample_download import get_sample

# 방법 1: 강사 제공 샘플 이미지 사용
img = cv2.imread('./img/pedestrian_02.jpg')

if img is None:
    print("Error: 이미지 로드 실패")

else:
    print(f"✅ 이미지 로드 성공: {img.shape}")
    cv2.imshow('Original', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ① HOG 디스크립터 생성 ---
hog = cv2.HOGDescriptor()

# ② 사전학습된 보행자 검출 모델 로드 ---
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# ③ 이미지에서 보행자 검출 ---
detections, weights = hog.detectMultiScale(
    img,
    winStride=(8, 8),      # 스캔 윈도우 이동 크기
    padding=(16, 16),      # 윈도우 주변 패딩
    scale=1.05             # 이미지 피라미드 스케일
)

print(f"검출된 보행자: {len(detections)}명")

# ④ 신뢰도 분석 ---
if len(weights) > 0:
    print(f"신뢰도 범위: {weights.min():.3f} ~ {weights.max():.3f}")
    print(f"신뢰도별 개수:")

    for threshold in [0.3, 0.5, 0.7, 0.9]:
        count = np.sum(weights > threshold)
        print(f"  weights > {threshold}: {count}명")

# ⑤ 검출 결과 시각화 ---
result = img.copy()

for (x, y, w, h) in detections:
    cv2.rectangle(result, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.imshow('Pedestrian Detection', result)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 신뢰도 필터링
CONFIDENCE_THRESHOLD = 0.5  # 임계값 조정 (0.3 ~ 0.9)
result_filtered = img.copy()
filtered_count = 0

for (x, y, w, h), weight in zip(detections, weights):
    if weight > CONFIDENCE_THRESHOLD:
        cv2.rectangle(result_filtered, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # 신뢰도 표시
        cv2.putText(result_filtered, f'{weight:.2f}', (x, y-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        filtered_count += 1

print(f"\n신뢰도 필터링 (threshold={CONFIDENCE_THRESHOLD}):")
print(f"  필터링 전: {len(detections)}명")
print(f"  필터링 후: {filtered_count}명")
print(f"  감소율: {(len(detections)-filtered_count)/len(detections)*100:.1f}%")

cv2.imshow('Filtered Detection', result_filtered)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 다양한 파라미터 조합으로 성능 비교
configs = [
    {"name": "Default", "winStride": (8, 8), "scale": 1.05},
    {"name": "Fine-grained", "winStride": (4, 4), "scale": 1.05},
    {"name": "Small objects", "winStride": (8, 8), "scale": 1.02},
    {"name": "Fast", "winStride": (16, 16), "scale": 1.1},
]

print("\n파라미터별 검출 결과:")
print("-" * 60)

for config in configs:
    found, _ = hog.detectMultiScale(
        img,
        winStride=config["winStride"],
        padding=(16, 16),
        scale=config["scale"]
    )
    
    display_img = img.copy()
    # 설정의 이름을 제목으로 표시
    cv2.putText(display_img, f"Mode: {config['name']}", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # 찾은 사람들에게 사각형 그리기
    for (x, y, w, h) in found:
        cv2.rectangle(display_img, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # 결과창 띄우기
    cv2.imshow(f"Result - {config['name']}", display_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print(f"{config['name']:15s} (winStride={config['winStride']}, scale={config['scale']}): {len(found):3d}명")
