import cv2

chars = ' .,-~:;=!*#$@'

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 영상 끝나면 처음으로
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (200, 50))

    result = ''
    for row in resized:
        for pixel in row:
            idx = min(int(pixel / 256 * len(chars)), len(chars) - 1)
            result += chars[idx]
        result += '\n'

    # ANSI 이스케이프 코드로 화면 지우고 다시 출력 (애니메이션 효과)
    print('\x1b[2J\x1b[H' + result)

cap.release()