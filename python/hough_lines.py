import cv2
import numpy as np
import matplotlib.pyplot as plt

src_img = cv2.imread('./images/quiet_street.png')
h, w = src_img.shape[:2]

src_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)
gray_img = cv2.cvtColor(src_img, cv2.COLOR_RGB2GRAY)
gray_img[:,480:] = cv2.GaussianBlur(gray_img[:,480:], (5, 5), 1.2)
edge_map = cv2.Canny(gray_img, 40, 120)

# standard Hough transform
lines = cv2.HoughLines(edge_map, 1, np.pi / 128, 125)
anno_img_SHT = src_img.copy()

for line in lines:
    rho, theta = line.flat
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 - h * w * b)
    y1 = int(y0 + h * w * a)
    x2 = int(x0 + h * w * b)
    y2 = int(y0 - h * w * a)
    cv2.line(anno_img_SHT, (x1, y1), (x2, y2), (0, 255, 0), 2)

# probabilistic Hough transform
lines = cv2.HoughLinesP(edge_map, 1, np.pi / 128, 75, minLineLength=60, maxLineGap=20)
anno_img_PHT = src_img.copy()

for line in lines:
    x1, y1, x2, y2 = line.flat
    cv2.line(anno_img_PHT, (x1, y1), (x2, y2), (0, 255, 0), 2)

plt.figure('Hough Lines', figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.imshow(src_img, cmap='gray')
plt.title('\"It Almost Seems Too Quiet\" Street')

plt.subplot(2, 2, 2)
plt.imshow(edge_map, cmap='gray')
plt.title('Edge')

plt.subplot(2, 2, 3)
plt.imshow(anno_img_SHT)
plt.title('Hough Lines')

plt.subplot(2, 2, 4)
plt.imshow(anno_img_PHT)
plt.title('Probabilistic Hough Lines')

plt.tight_layout()
plt.show()