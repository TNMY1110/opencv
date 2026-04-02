import cv2 as cv
import numpy as np
import urllib.request
import os
from pathlib import Path
import matplotlib.pylab as plt 


# ============================================================
# 전역 변수
# ============================================================
win_name = "Document Scanning"
img = None
draw = None
rows, cols = 0, 0
pts_cnt = 0
pts = np.zeros((4, 2), dtype=np.float32)

def get_sample(filename, repo='insightbook'):
    if not os.path.exists(filename):
        if repo == 'insightbook':
            url = f"https://raw.githubusercontent.com/dltpdn/insightbook.opencv_project_python/master/img/{filename}"
        else:  # opencv 공식
            url = f"https://raw.githubusercontent.com/opencv/opencv/master/samples/data/{filename}"
        urllib.request.urlretrieve(url, filename)
    return filename

def onMouse(event, x, y, flags, param):
    """
    마우스로 4개 점을 클릭하면:
    1. 클릭 위치에 초록색 원 표시
    2. 4개 점 수집 후 자동으로 좌상/우상/우하/좌하 판단
    3. 원근 변환 적용
    """
    global pts_cnt, draw, pts, img
    
    if event == cv.EVENT_LBUTTONDOWN:
        # 1️⃣ 클릭한 위치에 원 표시
        cv.circle(draw, (x, y), 5, (0,255,0), -1)

        # 2️⃣ 좌표 저장
        pts[pts_cnt] = [x, y]
        pts_cnt += 1
        
        cv.imshow("Scan", draw)

        # 3️⃣ 4개 점 수집 완료 → 좌표 정렬 + 변환
        if pts_cnt == 4:
            # 합 계산 (좌상단: 최소, 우하단: 최대)
            sum_pts = pts.sum(axis=1)
            # 차 계산 (우상단: 최소, 좌하단: 최대)
            diff_pts = np.diff(pts, axis=1)

            # 변환 전 4개 좌표
            pts1 = np.float32([
                    pts[np.argmin(sum_pts)],   # 좌상단
                    pts[np.argmin(diff_pts)],   # 우상단
                    pts[np.argmax(sum_pts)],   # 우하단
                    pts[np.argmax(diff_pts)],   # 좌하단
                ])

            # 변환 후 서류 크기 계산
            pts2 = np.float32([[0,0],[300,0],[300,300],[0,300]])
            M = cv.getPerspectiveTransform(pts1,pts2)
            dst = cv.warpPerspective(img,M,(300,300))

            # 원근 변환 적용
            plt.subplot(121),plt.imshow(cv.cvtColor(draw, cv.COLOR_BGR2RGB)),plt.title('Input')
            plt.subplot(122),plt.imshow(cv.cvtColor(dst, cv.COLOR_BGR2RGB)),plt.title('Output')
            plt.show()

            # 결과 저장
            cv.imwrite("paper_scan.png", dst)

            # 초기화 (새로운 이미지 스캔 가능)
            pts_cnt = 0
            pts = np.zeros((4, 2), dtype=np.float32)
            draw = img.copy()
            cv.imshow("Scan", draw)

# ============================================================
# 메인 실행
# ============================================================
img = cv.imread(get_sample('paper.jpg', repo='insightbook'))

if img is None:
    print("이미지를 불러올 수 없습니다.")
    exit()

draw = img.copy()
rows, cols = img.shape[:2]

# 윈도우 표시 + 마우스 콜백 등록
cv.imshow("Scan", draw)
cv.setMouseCallback('Scan', onMouse)

# print("📝 사용법:")
# print("1. 이미지 위에 4개 점을 클릭하세요 (좌상단, 우상단, 우하단, 좌하단 순서 무관)")
# print("2. 4번째 점 클릭 후 자동으로 문서 스캔이 실행됩니다.")
# print("3. 'Scanned Document' 윈도우에서 결과를 확인하세요.")
#
k = cv.waitKey(0)

cv.destroyAllWindows()
