import cv2 as cv
import urllib.request
import sys
import os

def get_sample(filename, flags=cv.IMREAD_COLOR):
    if not os.path.exists(filename):
        url = f"https://raw.githubusercontent.com/opencv/opencv/master/samples/data/{filename}"
        urllib.request.urlretrieve(url, filename)
    
    return cv.imread(filename, flags)

# Create a black image
def draw_name():
    img = np.zeros((512,512,3), np.uint8)

    cv.rectangle(img,(255,255),(512,512),(0,0,0),-1)

    font = cv.FONT_HERSHEY_SIMPLEX
    cv.putText(img,'Tony Stark',(10,500), font, 4,(255,255,255),2,cv.LINE_AA)

    cv.imshow("Drawing", img)
    cv.waitKey(0)
    cv.destroyAllWindows()

# img = cv.imread("./captures/starry_night.jpg")
# img_gray = cv.imread("./samples/starry_night.jpg", cv.IMREAD_GRAYSCALE)
filename = "orange.jpg"
basename = os.path.splitext(filename)[0]  # "starry_night"

img = get_sample(filename)
img_gray = get_sample(filename, cv.IMREAD_GRAYSCALE)

if img is None:
    sys.exit("Could not read the image.")

cv.imshow("Display window", img)

# 이미지 정보 확인
print("=== 컬러 이미지 ===")
print(f"shape: {img.shape}") # (높이, 너비, 채널)
print(f"shape: {img.dtype}") # 파일의 형태
print(f"shape: {img.size}") # 전체 픽셀 수(높이 * 너비 * 채널)

print("=== 그레이 스케일 이미지 ===")
print(f"shape: {img_gray.shape}") # (높이, 너비, 채널)
print(f"shape: {img_gray.dtype}") # 파일의 형태
print(f"shape: {img_gray.size}") # 전체 픽셀 수(높이 * 너비 * 채널)

# 창 닫기
k = cv.waitKey(0)

# 파일 저장
if k == ord("s"):
    cv.imwrite("./save/{basename}.png", img)
    cv.imwrite("./save/{basename}_gray.png", img_gray)