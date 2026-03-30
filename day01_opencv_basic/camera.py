import numpy as np
import cv2 as cv
import os

cap = cv.VideoCapture(0)
os.makedirs("./captures", exist_ok=True)  # 폴더 없으면 생성

def next_frame_count(folder):
    i = 0
    while os.path.exists(f"{folder}/capture_{i}.png"):
        i += 1
    return i

if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # Our operations on the frame come here
    # gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # Display the resulting frame
    cv.imshow('frame', frame)
    key = cv.waitKey(1)

    if key == ord('q'):
        break

    if key == ord('c'):
        frame_count = next_frame_count("./captures")
        filename = f"capture_{frame_count}.png"
        cv.imwrite(f"./captures/{filename}", frame)
        print(f"캡쳐 저장: {filename}")

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()