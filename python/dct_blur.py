import cv2
import numpy as np
import matplotlib.pyplot as plt

KERNEL_RADIUS = 300


src_img = cv2.imread('./images/testcard720.png')
h, w, c = src_img.shape

gray_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
gray_img = np.float32(gray_img) / 255.0

blur_filter = cv2.getGaussianKernel(
    KERNEL_RADIUS* 2, 0, ktype=cv2.CV_32F)[KERNEL_RADIUS:]
    
blur_filter /= blur_filter[0]

blur_filter_padded = np.zeros_like(gray_img)
blur_filter_padded[:KERNEL_RADIUS,:KERNEL_RADIUS] = \
    blur_filter @ blur_filter.T

dct_img = cv2.dct(gray_img)
filtered_img = cv2.idct(dct_img * blur_filter_padded)

ratio = np.max(gray_img) / np.max(filtered_img)
print(ratio)
print(h * w)

plt.figure('DCT Blur', figsize=(16, 5))

plt.subplot(1, 2, 1)
plt.imshow(gray_img, cmap='gray', vmin=0, vmax=1)

plt.subplot(1, 2, 2)
plt.imshow(filtered_img, cmap='gray', vmin=0, vmax=1)

plt.tight_layout()
plt.show()
