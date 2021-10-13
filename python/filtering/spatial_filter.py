import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils.image_reader import ImageReader

if __name__ == "__main__":
    reader = ImageReader()
    img_src = reader.read("images/lena.heic", np.float32)

    kernel = np.array([
        [-1, -1, -1],
        [-1, +8, -1],
        [-1, -1, -1]], dtype=np.float32)
    img_filtered = cv2.filter2D(img_src, cv2.CV_32F, kernel)

    # gaussian_kernel = cv2.getGaussianKernel(7, 2.0, ktype=cv2.CV_32F)
    # kernel_l = gaussian_kernel[:4]
    # kernel_l /= np.sum(kernel_l)
    # kernel_r = np.flip(kernel_l, axis=0)
    # print(kernel_l)
    # print(kernel_r)
    # img_filtered = cv2.sepFilter2D(img_src, cv2.CV_32F, kernel_l, kernel_r)

    fig, axs = plt.subplots(1, 2, num="Spatial filtering", figsize=(12, 6))
    axs[0].set_title("Source")
    axs[0].imshow(img_src)
    axs[1].set_title("Filtered")
    axs[1].imshow(img_filtered)

    plt.show()
