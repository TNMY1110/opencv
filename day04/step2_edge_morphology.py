import cv2 as cv
import numpy as np
import urllib.request
import os

def get_sample(filename, repo='insightbook'):
    if not os.path.exists(filename):
        if repo == 'insightbook':
            url = f"https://raw.githubusercontent.com/dltpdn/insightbook.opencv_project_python/master/img/{filename}"
        else:  # opencv 공식
            url = f"https://raw.githubusercontent.com/opencv/opencv/master/samples/data/{filename}"
        urllib.request.urlretrieve(url, f"./img/{filename}")
    return f"./img/{filename}"

# 이미지 로드
img = cv.imread(get_sample('moon_gray.jpg', repo='insightbook'))

if img is None:
    print("이미지를 불러올 수 없습니다.")
    exit()

# 그레이스케일 변환
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# ============================================================
# 1. Canny 에지 검출
# ============================================================
threshold1 = 50   # 낮은 임계값
threshold2 = 150  # 높은 임계값

# Canny 에지 검출 적용
edges = cv.Canny(gray, threshold1, threshold2)

# ============================================================
# 2. 모폴로지 연산 — 열기 (Opening)
# ============================================================
# 커널 생성 (5x5 타원)
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))

# 열기 연산 (침식 후 팽창: 노이즈 제거)
edges_cleaned = cv.morphologyEx(edges, cv.MORPH_OPEN, kernel)

# 선택사항: 닫기 연산 (팽창 후 침식: 구멍 채우기)
edges_closed = cv.morphologyEx(edges_cleaned, cv.MORPH_CLOSE, kernel)

# ============================================================
# 3. 결과 비교 표시
# ============================================================
# 원본 → Canny → 열기 → 닫기 순서로 4개 이미지 배열
canny_color = cv.cvtColor(edges, cv.COLOR_GRAY2BGR)
cleaned_color = cv.cvtColor(edges_cleaned, cv.COLOR_GRAY2BGR)
closed_color = cv.cvtColor(edges_closed, cv.COLOR_GRAY2BGR)

top_row = np.hstack([img, canny_color])
bottom_row = np.hstack([cleaned_color, closed_color])
result = np.vstack([top_row, bottom_row])

cv.imshow('Edge Detection + Morphology', result)
cv.waitKey(0)
cv.destroyAllWindows()
