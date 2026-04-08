import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import time

def draw_matches(img1, kp1, img2, kp2, matches, title="Feature Matching", mask=None):
    draw_params = dict(
            matchColor=(0, 255, 0),
            singlePointColor=None,
            matchesMask=mask,
            flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
    
    res = cv.drawMatches(img1, kp1, img2, kp2, matches, None, **draw_params)
    
    # Matplotlib 출력
    plt.figure(figsize=(15, 10))
    # 만약 입력 이미지가 그레이스케일이라면 컬러 출력을 위해 변환이 필요할 수 있음
    if len(res.shape) == 3:
        plt.imshow(cv.cvtColor(res, cv.COLOR_BGR2RGB))
    else:
        plt.imshow(res, cmap='gray')
        
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

img_logo = cv.imread('./img/benz.png', cv.IMREAD_GRAYSCALE)
img_scene = cv.imread('./img/benz_drive_01.jpg')
gray_scene = cv.cvtColor(img_scene, cv.COLOR_BGR2GRAY)

if img_logo is None or img_scene is None:
    print("Error: 이미지를 찾을 수 없습니다.")
    exit()

h, w = img_logo.shape[:2] # 로고의 높이와 너비 가져오기

logo_mask = np.zeros(img_logo.shape[:2], dtype=np.uint8)
cv.circle(logo_mask, (w//2, h//2), int(w*0.4), 255, -1)
cv.imshow('Logo Mask', logo_mask)
combined_view = cv.bitwise_and(img_logo, img_logo, mask=logo_mask)
cv.imshow('Is the logo inside the circle?', combined_view)

# 2) 특징점 검출기 초기화 (품질을 위해 SIFT 사용)
sift = cv.SIFT_create()

# 로고의 특징점 미리 추출
kp_logo, des_logo = sift.detectAndCompute(img_logo, logo_mask)

# --- 처리 시간 측정 시작 ---
start_time = time.time()

# 3) 현재 장면에서 특징점 추출
kp_scene, des_scene = sift.detectAndCompute(gray_scene, None)

# 4) FLANN 매칭
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)

flann = cv.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des_logo, des_scene, k=2)

# 5) Lowe's 비율 테스트 (품질 우선: 0.7)
good_matches = []

for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good_matches.append(m)

# 6) 호모그래피 및 위치 표시
MIN_MATCH_COUNT = 10

result_img = img_scene.copy()
detected = False

if len(good_matches) >= MIN_MATCH_COUNT:
    src_pts = np.float32([kp_logo[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_scene[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 3.0)
    
    if M is not None:
        h, w = img_logo.shape
        pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
        dst = cv.perspectiveTransform(pts, M)
        
        # 바운딩 박스 그리기 (파란색)
        cv.polylines(result_img, [np.int32(dst)], True, (255, 0, 0), 3, cv.LINE_AA)

        matchesMask = mask.ravel().tolist()
        draw_matches(img_logo, kp_logo, result_img, kp_scene, good_matches, title="Benz Logo - Inlier Matches Only", mask=matchesMask)
        
        detected = True

# --- 처리 시간 측정 종료 ---
end_time = time.time()
process_time = (end_time - start_time) * 1000 # ms 단위

# 7) 결과 출력
status = "Success" if detected else "Failed"
print(f"[{status}] Time: {process_time:.2f}ms, Matches: {len(good_matches)}")

plt.figure(figsize=(10, 8))
plt.imshow(cv.cvtColor(result_img, cv.COLOR_BGR2RGB))
plt.title(f"Detection: {status} ({process_time:.1f}ms)")
plt.axis('off')
plt.show()
