import cv2

chars = ' .,-~:;=!*#$@' # 밝음 -> 어둠

img = cv2.imread('./img/jungsanghwa.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

h, w = gray.shape
nw = 100
nh = int(h/w * nw)

gray = cv2.resize(gray, (nw * 2, nh))

for row in gray:
    for pixel in row: # pixel 0-255 -> CHARS 0-12
        index = int(pixel / 256 * len(chars))
        print(chars[index], end='')

    print()