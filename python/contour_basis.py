import cv2
import numpy as np
import matplotlib.pyplot as plt

src_img = cv2.imread('./images/contour_test.png')
gray_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
h, w = src_img.shape[:2]

gray_img = cv2.medianBlur(gray_img, 5)
_, mask = cv2.threshold(gray_img, 150, 255, cv2.THRESH_BINARY_INV)

strel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, strel)

contours, hierarchy = cv2.findContours(
    mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours(src_img, contours, -1, (255, 0, 0), 2, lineType=cv2.LINE_AA)

plt.figure('Contour', figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(src_img)
plt.title('Original')

plt.subplot(1, 2, 2)
plt.imshow(mask, cmap='gray')
plt.title('Mask')

plt.tight_layout()
plt.show()