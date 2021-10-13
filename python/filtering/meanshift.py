import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils.image_reader import ImageReader

if __name__ == "__main__":
    reader = ImageReader()
    src_img = reader.read("images/dont_you_ever_stop.heic")
    src_img = cv2.resize(src_img, None, fx=0.5, fy=0.5,
                        interpolation=cv2.INTER_AREA)
    h, w = src_img.shape[:2]

    mean_shift_img = cv2.pyrMeanShiftFiltering(src_img, 15, 40, maxLevel=3)
    # edge = cv2.Canny(mean_shift_img, 200, 400)
    gray_img = cv2.cvtColor(mean_shift_img, cv2.COLOR_RGB2GRAY)

    # DoG
    # g1 = cv2.getGaussianKernel(7, 0.5, ktype=cv2.CV_32F)
    # g2 = cv2.getGaussianKernel(7, 1.5, ktype=cv2.CV_32F)
    # g = g1 - g2
    # LoG = cv2.sepFilter2D(gray_img, cv2.CV_16S, g, g)

    LoG = cv2.Laplacian(gray_img, cv2.CV_16S)
    minLoG = cv2.morphologyEx(LoG, cv2.MORPH_ERODE, np.ones((3, 3)))
    maxLoG = cv2.morphologyEx(LoG, cv2.MORPH_DILATE, np.ones((3, 3)))
    edge = np.logical_and(maxLoG > 40, LoG < -40)

    plt.figure("Flood Fill", figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(src_img)
    plt.title("Original")

    plt.subplot(1, 3, 2)
    plt.imshow(mean_shift_img)
    plt.title("Mean Shift")

    plt.subplot(1, 3, 3)
    plt.imshow(edge, cmap="gray")
    plt.title("Edge")

    plt.tight_layout()
    plt.show()
