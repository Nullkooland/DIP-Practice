import cv2
import numpy as np
import matplotlib.pyplot as plt

src_img = cv2.imread('./images/monarch.png')
src_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)
gray_img = cv2.cvtColor(src_img, cv2.COLOR_RGB2GRAY)

grad_x = cv2.Scharr(gray_img, cv2.CV_32F, 1, 0)
grad_y = cv2.Scharr(gray_img, cv2.CV_32F, 0, 1)

grad_mag, grad_angle = cv2.cartToPolar(grad_x, grad_y)

grad_mag = cv2.normalize(grad_mag, None, 0, 255,
                         norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
grad_angle = cv2.normalize(grad_angle, None, 0, 255,
                           norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

grad_img = cv2.merge((grad_angle, np.full_like(
    grad_mag, 255, dtype=np.uint8), grad_mag))
grad_img = cv2.cvtColor(grad_img, cv2.COLOR_HSV2RGB_FULL)

plt.figure('Gradient', figsize=(16, 6))

h, w, c = src_img.shape

plt.subplot(1, 2, 1)
plt.imshow(src_img)
plt.title('Original')

plt.subplot(1, 2, 2)
plt.imshow(grad_img)
plt.title('Gradient')

plt.tight_layout()
plt.show()
