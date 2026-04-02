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

cap = cv.VideoCapture(0)

def get_sample(filename, repo='insightbook'):
    if not os.path.exists(filename):
        if repo == 'insightbook':
            url = f"https://raw.githubusercontent.com/dltpdn/insightbook.opencv_project_python/master/img/{filename}"
        else:  # opencv 공식
            url = f"https://raw.githubusercontent.com/opencv/opencv/master/samples/data/{filename}"
        urllib.request.urlretrieve(url, filename)
    return filename

def onMouse(event, x, y, flags, param):
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
            cv.imwrite("cam_scan.png", dst)

            # 초기화 (새로운 이미지 스캔 가능)
            pts_cnt = 0
            pts = np.zeros((4, 2), dtype=np.float32)
            draw = img.copy()
            cv.imshow("Scan", draw)

# ============================================================
# 메인 실행
# ============================================================
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # 프레임 크기 조정 (보기 좋게)
    frame = cv.resize(frame, (800, 600))
    img = frame.copy()
    draw = frame.copy()
    
    cv.imshow(win_name, draw)
    cv.setMouseCallback(win_name, onMouse)
    
    if cv.waitKey(1) & 0xFF == ord('q'):  # 'q' 누르면 종료
        break

cap.release()
cv.destroyAllWindows()

