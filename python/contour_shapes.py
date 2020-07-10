import cv2
import numpy as np
import matplotlib.pyplot as plt

src_img = cv2.imread('./images/oil_droplet.png')
h, w = src_img.shape[:2]

src_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)
lab_img = cv2.cvtColor(src_img, cv2.COLOR_RGB2LAB)
mask = cv2.inRange(lab_img, (100, 140, 160), (240, 160, 210))

strel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, strel)


num_cc, cc = cv2.connectedComponentsWithAlgorithm(mask, 4, ltype=cv2.CV_32S, ccltype=cv2.CCL_WU)

plt.figure('Contour', figsize=(16, 6))

plt.subplot(1, 3, 1)
plt.imshow(src_img)
plt.title('Original')

plt.subplot(1, 3, 2)
plt.imshow(mask, cmap='gray')
plt.title('Mask')

plt.subplot(1, 3, 3)
plt.imshow(cc, cmap='magma')
plt.title('Connected Components')

plt.tight_layout()
plt.show()