import cv2
import numpy as np
import matplotlib.pyplot as plt
import pytesseract
#plt.style.use('dark_background')

path = './img/01가0785.jpg'
img_ori = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)

height, width, channel = img_ori.shape

plt.figure(figsize=(12, 10))
plt.imshow(img_ori, cmap='gray')
plt.show()
