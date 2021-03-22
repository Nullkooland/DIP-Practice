import cv2
import pyheif
import numpy as np
import pyheif
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable

src_img = pyheif.read_as_numpy("./images/monarch.heic")
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

fig, (ax0, ax1) = plt.subplots(1, 2, num="Gradient", figsize=(16, 6))

ax0.imshow(src_img)
ax0.set_title("Original")

ax1.imshow(grad_img)
ax1.set_title("Gradient")

divider = make_axes_locatable(ax1)
cax = divider.append_axes("bottom", size="4%", pad=0.1)

fig.colorbar(ScalarMappable(norm=Normalize(vmin=0, vmax=360), cmap="hsv"),
             cax=cax, label=r"gradient angle / $^\circ$", orientation="horizontal",
             ticks=np.arange(0, 361, 30))

plt.show()
