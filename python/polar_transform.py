import cv2
import pyheif
import numpy as np
import matplotlib.pyplot as plt
import math

img = np.zeros((512, 512, 3), dtype=np.uint8)
SIZES = [2, 4, 8, 16, 32, 64, 128, 160, 192, 224, 240, 248, 252, 254]
M = 80

for size in SIZES:
    pt0 = (256 - size, 256 - size)
    pt1 = (256 + size, 256 + size)
    img = cv2.rectangle(img, pt0, pt1,
                        (10, 155, 255), 1, lineType=cv2.LINE_AA)

    img = cv2.circle(img, (256, 256), size,
                     (200, 125, 0), 1, lineType=cv2.LINE_AA)

max_range = int(256 * math.sqrt(2))
polar_img = cv2.warpPolar(img, (512, 512), (256, 256),
                          max_range, cv2.INTER_CUBIC)

log_max_range = M * math.log(max_range)
log_polar_img = cv2.logPolar(img, (256, 256), M, cv2.INTER_CUBIC)

plt.figure("Polar Mapping", figsize=(16, 6))

plt.subplot(1, 3, 1)
plt.imshow(img)

plt.title("Original")
plt.xlabel(r"$x$")
plt.ylabel(r"$y$")
plt.xticks(np.arange(0, 513, 64))
plt.yticks(np.arange(0, 513, 64))

plt.subplot(1, 3, 2)
plt.imshow(polar_img, extent=[0, max_range, 360, 0], aspect=max_range/360)

plt.title("Polar")
plt.xlabel(r"$\rho$")
plt.ylabel(r"$\theta$")
plt.xticks(np.arange(0, max_range, 32))
plt.yticks(np.arange(0, 361, 45))

plt.subplot(1, 3, 3)
plt.imshow(log_polar_img, extent=[0, 512, 360, 0], aspect=512/360) # this is why matplotlib sucks

plt.title("Log Polar")
plt.xlabel(fr"{M} $\ln\rho$")
plt.ylabel(r"$\theta$")
plt.xticks(np.arange(0, log_max_range, 50))
plt.yticks(np.arange(0, 361, 45))

plt.tight_layout()
plt.show()
