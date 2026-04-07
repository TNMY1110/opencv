import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from sample_download import get_sample

def draw_matches(img1, kp1, img2, kp2, matches, title="Feature Matching", mask=None, draw_params=None):
    draw_params = dict(
        matchColor=(0, 255, 0),        # 매칭 선 색상 (초록색)
        singlePointColor=(255, 0, 0),  # 매칭되지 않은 특징점 색상 (빨간색)
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

# ========== Step 1: 이미지 로드 ==========
# 공식 샘플 이미지 또는 자신의 이미지 사용
img1 = cv.imread('./img/Mach_Deep.jpg', cv.IMREAD_GRAYSCALE)
img2 = cv.imread('./img/mach_deep_photo.jpg', cv.IMREAD_GRAYSCALE)

if img1 is None or img2 is None:
    print("Error: 이미지를 찾을 수 없습니다.")
    exit()

print(f"img1 shape: {img1.shape}, img2 shape: {img2.shape}")

# ========== Step 2: 특징점 검출기 초기화 ==========
# TODO: 아래 중 하나를 선택해서 코드 작성하세요
sift = cv.SIFT_create()
# sift = cv.ORB_create()

# ========== Step 3: 키포인트와 디스크립터 추출 ==========
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)
# 위 두 줄을 구현하세요
print(f"Keypoints found - img1: {len(kp1)}, img2: {len(kp2)}")

# ========== Step 4: FLANN 매칭기 설정 ==========
# SIFT는 float descriptor → KDTREE
# ORB는 binary descriptor → LSH
# TODO: detector_type을 판단하고 FLANN 파라미터 설정하세요
# SIFT 사용 시:
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
# ORB 사용 시:
#   FLANN_INDEX_LSH = 6
#   index_params = dict(algorithm=FLANN_INDEX_LSH, 
#                       table_number=12, key_size=20, multi_probe_level=2)
#   search_params = dict(checks=50)
flann = cv.FlannBasedMatcher(index_params, search_params)

# ========== Step 5: knnMatch로 k=2 매칭 ==========
matches = flann.knnMatch(des1, des2, k=2)
print(f"Total matches: {len(matches)}")

# ========== Step 6: Lowe's 비율 테스트 ==========
good_matches = []

for match_pair in matches:
    if len(match_pair) == 2:
        m, n = match_pair
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

print(f"Good matches after Lowe's ratio test: {len(good_matches)}")

# ========== Step 7: 시각화 ==========
# if len(good_matches) >= 10:
#     draw_matches(img1, kp1, img2, kp2, good_matches,
#                  title=f"Good Matches ({len(good_matches)})")
# else:
#     print("Not enough good matches!")



# ========== Step 1: 호모그래피 계산 ==========
MIN_MATCH_COUNT = 10

if len(good_matches) >= MIN_MATCH_COUNT:
    # TODO: 다음을 구현하세요
    # 1) good_matches에서 queryIdx와 trainIdx로 키포인트 좌표 추출
    # 2) np.float32 배열로 변환 (reshape(-1, 1, 2) 형태)
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    # TODO: 호모그래피 행렬 계산 (RANSAC 사용)
    M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 1.0)
    
    if M is not None:
        # TODO: img1의 네 모서리를 이미지 좌표 배열로 정의
        h, w = img1.shape
        pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
        
        # TODO: perspectiveTransform으로 변환된 좌표 계산
        dst = cv.perspectiveTransform(pts, M)
        
        # ========== Step 2: 결과 시각화 ==========
        # TODO: 다음을 구현하세요
        # 1) img2를 카피
        result_img = cv.cvtColor(img2, cv.COLOR_GRAY2BGR)
        # 2) cv.polylines()로 변환된 사각형 그리기 (파란색, 두께 3)
        cv.polylines(result_img, [np.int32(dst)], True, (255, 0, 0), 3, cv.LINE_AA)
        # 3) matplotlib으로 표시
        
        # TODO: polylines 코드 작성
        
        plt.figure(figsize=(10, 8))
        plt.imshow(cv.cvtColor(result_img, cv.COLOR_BGR2RGB))
        plt.title('Detected Object with Homography')
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        
        # ========== Step 3: 매칭 시각화 (inlier만) ==========
        matchesMask = mask.ravel().tolist()
        draw_params = dict(
            matchColor=(0, 255, 0),
            singlePointColor=None,
            matchesMask=matchesMask,
            flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )

        draw_matches(img1, kp1, result_img, kp2, good_matches, title="After Lowe's Ratio Test", mask=matchesMask)

        # TODO: drawMatches로 inlier만 표시
        inlier_count = sum(matchesMask)
        outlier_count = len(matchesMask) - inlier_count
        print(f"Inliers: {inlier_count}, Outliers: {outlier_count}")

    else:
        print("Failed to compute homography")
else:
    print(f"Not enough matches ({len(good_matches)}/{MIN_MATCH_COUNT})")
