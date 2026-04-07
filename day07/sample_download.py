import urllib.request
import os

def get_sample(filename):
    """OpenCV 공식 샘플 이미지 자동 다운로드"""
    if not os.path.exists(filename):
        url = f"https://raw.githubusercontent.com/opencv/opencv/master/samples/data/{filename}"
        urllib.request.urlretrieve(url, f"./img/{filename}")
    return filename
