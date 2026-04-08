import urllib.request
import os

def get_sample(filename, repo='opencv'):
    """OpenCV 공식 샘플 이미지 자동 다운로드"""
    if not os.path.exists(filename):
        if repo == 'insightbook':
            url = f"https://raw.githubusercontent.com/dltpdn/insightbook.opencv_project_python/master/img/{filename}"
        else:  # opencv 공식
            url = f"https://raw.githubusercontent.com/opencv/opencv/master/samples/data/{filename}"
        urllib.request.urlretrieve(url, f"./img/{filename}")
    return f"./img/{filename}"

get_sample('4027.png', repo='insightbook')