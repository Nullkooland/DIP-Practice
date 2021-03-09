import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

# Di Zenzo's multichannel gradient for RGB color images
# see Gonzalez & Woods, Digital Image Processing, 3rd Edition Chapter 6

src_img = cv2.imread('./images/cloth.png')
src_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)

# calculate gradient on x and y axis for RGB respectively

# R
grad_r_x = cv2.Scharr(src_img[..., 0], cv2.CV_32F, 1, 0)
grad_r_y = cv2.Scharr(src_img[..., 0], cv2.CV_32F, 0, 1)
# G
grad_g_x = cv2.Scharr(src_img[..., 1], cv2.CV_32F, 1, 0)
grad_g_y = cv2.Scharr(src_img[..., 1], cv2.CV_32F, 0, 1)
# B
grad_b_x = cv2.Scharr(src_img[..., 2], cv2.CV_32F, 1, 0)
grad_b_y = cv2.Scharr(src_img[..., 2], cv2.CV_32F, 0, 1)

g_xx = grad_r_x ** 2 + grad_g_x ** 2 + grad_b_x ** 2
g_yy = grad_r_y ** 2 + grad_g_y ** 2 + grad_b_y ** 2
g_xy = grad_r_x * grad_r_y + grad_g_x * grad_g_y + grad_b_x * grad_b_y

# there's ambiguity of gradient angle in this formula...
theta = np.arctan2(2 * g_xy, g_yy - g_xx)

grad_mag = cv2.sqrt(0.5 *
                    ((g_xx + g_yy) + (g_yy - g_xx) * np.cos(theta) +
                     2 * g_xy * np.sin(theta)))

grad_mag = cv2.normalize(grad_mag, None, 0, 255,
                         norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
theta = cv2.normalize(theta, None, 0, 255,
                           norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

grad_img = cv2.merge((theta, np.full_like(
    grad_mag, 255, dtype=np.uint8), grad_mag))
grad_img = cv2.cvtColor(grad_img, cv2.COLOR_HSV2RGB_FULL)

fig, (ax0, ax1) = plt.subplots(1, 2, num='Gradient', figsize=(14, 6))

ax0.imshow(src_img)
ax0.set_title('Original')

ax1.imshow(grad_img)
ax1.set_title('Gradient')
fig.colorbar(ScalarMappable(norm=Normalize(vmin=0, vmax=180), cmap='hsv'),
             ax=ax1, label='gradient angle',
             ticks=np.arange(0, 181, 30))

plt.tight_layout()
plt.show()
