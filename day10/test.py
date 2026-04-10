import cv2

for i in range(5):
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
    if cap.isOpened():
        print(f"{i}번 카메라 사용 가능")
        cap.release()
    else:
        print(f"{i}번 카메라 없음")