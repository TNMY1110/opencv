import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from sample_download import get_sample

# ========== Step 1: 정지 표지판 이미지 로드 ==========
# 자신의 이미지 경로를 입력하세요
img = cv.imread('./img/stop_sign.jpg')  # TODO: 파일 경로 수정

if img is None:
    print("Image not found!")
    exit()

hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
h, w = img.shape[:2]

# ========== Step 2: 빨간색 마스크 생성 ==========
# 빨강: H 0~10, 170~180 (HSV에서 H는 0~180)
# TODO: 두 개의 빨간색 범위 마스크 생성 후 OR 연산
# lower_red1, upper_red1 정의
lower_red1 = np.array([0, 50, 50])
upper_red1 = np.array([10, 255, 255])

# lower_red2, upper_red2 정의
lower_red2 = np.array([170, 50, 50])
upper_red2 = np.array([179, 255, 255])

mask1 = cv.inRange(hsv, lower_red1, upper_red1)
mask2 = cv.inRange(hsv, lower_red2, upper_red2)
red_mask = cv.bitwise_or(mask1, mask2)
print(f"Red pixels: {cv.countNonZero(red_mask)}")

# ========== Step 3: 노이즈 제거 (모폴로지) ==========
# TODO: 아래 코드를 구현하세요
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
red_mask = cv.morphologyEx(red_mask, cv.MORPH_CLOSE, kernel)
red_mask = cv.morphologyEx(red_mask, cv.MORPH_OPEN, kernel)

# ========== Step 4: 컨투어 검출 ==========
# TODO: cv.findContours() 사용
# O와 P 안의 빨간 부분까지 표지판으로 인식해서 RETR_TREE대신 RETR_EXTERNAL 사용
contours, _ = cv.findContours(red_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE) 
print(f"Found {len(contours)} contours")

# ========== Step 5: 컨투어 필터링 (면적 및 형태) ==========
min_area = 100
detected_signs = []
for contour in contours:
    area = cv.contourArea(contour)
    if area < min_area:
        continue
    
    # TODO: 다음을 구현하세요
    # 1) 근사 다각형(approxPolyDP) 계산
    perimeter = cv.arcLength(contour, True)
    approx = cv.approxPolyDP(contour, 0.02 * perimeter, True)
    
    # 2) 꼭짓점 개수 확인 (8각형은 8개)
    num_vertices = len(approx)
    
    # 3) 바운딩 박스의 aspect ratio 계산
    x, y, w, h = cv.boundingRect(contour)
    aspect_ratio = float(w) / h if h > 0 else 0
    
    # 4) 정사각형 형태 확인 (aspect ratio 0.8~1.2)
    if 0.8 <= aspect_ratio <= 1.2:
        if 7 <= num_vertices <= 10:  # 팔각형(8)을 기준으로 앞뒤 여유를 둡니다.
            detected_signs.append((x, y, w, h, num_vertices))

    # 5) detected_signs 리스트에 추가
    if num_vertices >= 6:  # 최소 6개 꼭짓점 (팔각형도 충분히 감지)
        if 0.8 <= aspect_ratio <= 1.2:
            detected_signs.append((x, y, w, h, num_vertices))

print(f"Detected stop signs: {len(detected_signs)}")

# ========== Step 6: 결과 시각화 ==========
result_img = img.copy()

for x, y, w, h, vertices in detected_signs:
    # TODO: 바운딩 박스와 텍스트 그리기
    cv.rectangle(result_img, (x, y), (x+w, y+h), (0, 0, 255), 2)
    cv.putText(result_img, f'Stop ({vertices}v)', (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(cv.cvtColor(red_mask, cv.COLOR_GRAY2RGB))
plt.title('Red Color Mask')
plt.axis('off')
plt.subplot(122)
plt.imshow(cv.cvtColor(result_img, cv.COLOR_BGR2RGB))
plt.title(f'Detected Stop Signs ({len(detected_signs)})')
plt.axis('off')
plt.tight_layout()
plt.show()
