import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from utils.image_reader import ImageReader

N_LEVEL = 5

if __name__ == "__main__":
    reader = ImageReader()
    src_img = reader.read("images/spike.heic", np.float32)

    gaussian_pyramid = [src_img]
    laplacian_pyramid = []

    for i in range(N_LEVEL):
        gaussian_pyramid.append(cv2.pyrDown(gaussian_pyramid[i]))
        laplacian_pyramid.append(
            gaussian_pyramid[i] - cv2.pyrUp(gaussian_pyramid[i + 1]))

    plt.figure("Pyramid", figsize=(16, 6))

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
