import numpy as np
import cv2
from sample_download import get_sample

# ① 이미지 로드 ---
K = 16  # 클러스터 개수 (16가지 색상으로 압축)

img = cv2.imread(get_sample('taekwonv1.jpg', repo='insightbook'))
# ② 데이터 형식 변환 ---
# 이미지를 "행x열x3(RGB)" 형태에서 "픽셀 수 x 3" 형태로 변환
data = img.reshape((-1, 3)).astype(np.float32)
# ③ 반복 중지 조건 정의 ---
# 최대 10번 반복하거나 오차가 1.0 이하가 되면 멈추기
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
# ④ k-means 실행 ---
ret, label, center = cv2.kmeans(data, K, None, criteria, 11, cv2.KMEANS_RANDOM_CENTERS)
# ⑤ 중심값(대표 색상)을 정수형으로 변환 ---
center = np.uint8(center)
print("중심 색상(BGR):")
print(center)
# ⑥ 각 픽셀을 해당 클러스터의 중심값으로 변환 ---
# label에는 각 픽셀이 어느 클러스터에 속하는지 저장됨
res = center[label.flatten()]
# ⑦ 원본 이미지 형태로 변환 ---
res = res.reshape(img.shape)
# ⑧ 원본과 압축 이미지 비교 ---
merged = np.hstack((img, res))
cv2.imshow('K-Means Color Compression', merged)
cv2.waitKey(0)
cv2.destroyAllWindows()
