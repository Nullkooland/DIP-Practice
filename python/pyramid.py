import cv2
import pyheif
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

N_LEVEL = 5

src_img = pyheif.read_as_numpy("./images/spike.heic")
src_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)
src_img = np.float32(src_img) / 255.0

gaussian_pyramid = [src_img]
laplacian_pyramid = []

for i in range(N_LEVEL):
    gaussian_pyramid.append(cv2.pyrDown(gaussian_pyramid[i]))
    laplacian_pyramid.append(
        gaussian_pyramid[i] - cv2.pyrUp(gaussian_pyramid[i + 1]))

plt.figure("Pyramid", figsize=(12, 6))

for i, pyr in enumerate(gaussian_pyramid):
    plt.subplot(2, N_LEVEL, i + 1)
    plt.imshow(pyr)

for i, pyr in enumerate(laplacian_pyramid):
    plt.subplot(2, N_LEVEL, i + N_LEVEL + 1)
    plt.imshow(pyr)

plt.tight_layout()


recovered_img = laplacian_pyramid[0] + cv2.pyrUp(gaussian_pyramid[1])

plt.figure("Spike is happy", figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(src_img)
plt.subplot(1, 2, 2)
plt.imshow(recovered_img)

plt.tight_layout()
plt.show()
