import cv2
import pyheif
import numpy as np
import matplotlib.pyplot as plt

# src_img = pyheif.read_as_numpy("./images/corn_kernels.heic")
# mask = src_img[..., 3]

# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
# mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

# cv2.imshow("corns", src_img)
# cv2.imshow("mask", mask)

# mask = ~mask

mask = np.ones((512, 512), dtype=np.uint8)

mask[0, 511] = 0
mask[256, 256] = 0
mask[128, 128] = 0
mask[480, 80] = 0

dist_l2 = cv2.distanceTransform(mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
dist_l1 = cv2.distanceTransform(mask, cv2.DIST_L1, cv2.DIST_MASK_3)
dist_linf = cv2.distanceTransform(mask, cv2.DIST_C, cv2.DIST_MASK_5)

plt.figure("Distance Transform", figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(dist_l2, cmap="inferno")
plt.title(r"$L_2$" + " (Euclidean Distance)")

plt.subplot(1, 3, 2)
plt.imshow(dist_l1, cmap="inferno")
plt.title(r"$L_1$" + " (City Distance)")

plt.subplot(1, 3, 3)
plt.imshow(dist_linf, cmap="inferno")
plt.title(r"$L_\infty$" + " (Checkerboard Distance)")

plt.tight_layout()
plt.show()