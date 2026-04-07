import cv2, numpy as np
from sample_download import get_sample

img1 = cv2.imread(get_sample('taekwonv1.jpg', repo='insightbook'))
img2 = cv2.imread(get_sample('figures.jpg', repo='insightbook'))
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# ORB로 서술자 추출 ---①
detector = cv2.ORB_create()
kp1, desc1 = detector.detectAndCompute(gray1, None)
kp2, desc2 = detector.detectAndCompute(gray2, None)
# BF-Hamming으로 매칭 ---②
#matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matcher = cv2.BFMatcher(cv2.NORM_HAMMING2)
# knnMatch, k=2
matches = matcher.knnMatch(desc1, desc2,2)

# 매칭 결과를 거리기준 오름차순으로 정렬 ---③
#matches = sorted(matches, key=lambda x:x.distance)
# 최소 거리 값과 최대 거리 값 확보 ---④
#min_dist, max_dist = matches[0].distance, matches[-1].distance
# 최소 거리의 20% 지점을 임계점으로 설정 ---⑤
ratio = 0.75
#ratio = 0.8
#good_thresh = (max_dist - min_dist) * ratio + min_dist
# 임계점 보다 작은 매칭점만 좋은 매칭점으로 분류 ---⑥
#good_matches = [m for m in matches if m.distance < good_thresh]
#print('matches:%d/%d, min:%.2f, max:%.2f, thresh:%.2f' \
        #%(len(good_matches),len(matches), min_dist, max_dist, good_thresh))
good_matches = [first for first,second in matches \
                    if first.distance < second.distance * ratio]
print('matches:%d/%d' %(len(good_matches),len(matches)))

# 좋은 매칭점만 그리기 ---⑦
res = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, \
                flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
# 결과 출력
cv2.imshow('Match', res)
cv2.waitKey()
cv2.destroyAllWindows()