import cv2
import numpy as np

CHARS = ' .,-~:;=!*#$@' # 13
nw = 100
font = cv2.FONT_HERSHEY_PLAIN
font_scale = 0.7
char_w, char_h = 7, 12

cap = cv2.VideoCapture(0)

print("\x1b[2J", end='')

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        break

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    h, w = img.shape
    nh = int(h / w * nw)

    img = cv2.resize(img, (nw * 2, nh))

    # 출력 캔버스 생성
    canvas = np.zeros((nh * char_h, nw * 2 * char_w), dtype=np.uint8)

    for y, row in enumerate(img):
        for x, pixel in enumerate(row):
            index = int(pixel / 256 * len(CHARS))
            char = CHARS[index]
            cv2.putText(canvas, char, (x * char_w, (y + 1) * char_h),
                        font, font_scale, 255, 1)

    cv2.imshow('ASCII Art', canvas)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
    # for row in img:
    #     for pixel in row: # pixel 0-255 -> CHARS 0-12
    #         index = int(pixel / 256 * len(CHARS))
    #         print(CHARS[index], end='')

    #     print()

    # print('\x1b[H', end='')