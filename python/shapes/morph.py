import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils.image_reader import ImageReader

if __name__ == "__main__":
    reader = ImageReader()
    img_src = reader.read("images/neon_flowers.heic")

    img_gray = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)

    # kernel_y = np.array([3, 10, 3], dtype=np.int16)
    # kernel_x = np.array([-1, 0, 1], dtype=np.int16)
    # grad_img = cv2.sepFilter2D(gray_img, cv2.CV_8U, kernel_x, kernel_y, delta=127)

    # cv2.imshow("gradient", grad_img)
    # cv2.waitKey()

    ret, mask = cv2.threshold(img_gray, 100, 255, cv2.THRESH_OTSU)
    # mask_img = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 17, 5)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernel)

    img_masked = cv2.copyTo(img_src, mask)
    img_masked = cv2.bilateralFilter(img_masked, 7, 3.4, 45)

    fig, axs = plt.subplots(1, 3, figsize=(12, 5))

    axs[0].imshow(img_src)
    axs[0].set_title("Source")

    axs[1].imshow(mask)
    axs[1].set_title("Mask")

    axs[2].imshow(img_masked)
    axs[2].set_title("Result")

    plt.show()
