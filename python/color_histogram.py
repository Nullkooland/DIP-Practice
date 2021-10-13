import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils.image_reader import ImageReader

if __name__ == "__main__":
    reader = ImageReader()
    img_rgb = reader.read("images/hand.heic", ignore_alpha=False)
    mask = img_rgb[..., 3] if img_rgb.shape[2] == 4 else None
    h, w = img_rgb.shape[:2]

    # RGB
    hist_red = cv2.calcHist([img_rgb], [0], mask, [256], [0, 256])
    hist_green = cv2.calcHist([img_rgb], [1], mask, [256], [0, 256])
    hist_blue = cv2.calcHist([img_rgb], [2], mask, [256], [0, 256])

    hist_red = cv2.normalize(hist_red, 0, 1, norm_type=cv2.NORM_L1)
    hist_green = cv2.normalize(hist_green, 0, 1, norm_type=cv2.NORM_L1)
    hist_blue = cv2.normalize(hist_blue, 0, 1, norm_type=cv2.NORM_L1)

    # HSV
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV_FULL)

    hist_h = cv2.calcHist([img_hsv], [0], mask, [256], [0, 256])
    hist_s = cv2.calcHist([img_hsv], [1], mask, [256], [0, 256])
    hist_v = cv2.calcHist([img_hsv], [2], mask, [256], [0, 256])

    hist_h = cv2.normalize(hist_h, 0, 1, norm_type=cv2.NORM_L1)
    hist_s = cv2.normalize(hist_s, 0, 1, norm_type=cv2.NORM_L1)
    hist_v = cv2.normalize(hist_v, 0, 1, norm_type=cv2.NORM_L1)

    # Lab
    img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)

    hist_l = cv2.calcHist([img_lab], [0], mask, [256], [0, 256])
    hist_a = cv2.calcHist([img_lab], [1], mask, [256], [0, 256])
    hist_b = cv2.calcHist([img_lab], [2], mask, [256], [0, 256])

    hist_l = cv2.normalize(hist_l, 0, 1, norm_type=cv2.NORM_L1)
    hist_a = cv2.normalize(hist_a, 0, 1, norm_type=cv2.NORM_L1)
    hist_b = cv2.normalize(hist_b, 0, 1, norm_type=cv2.NORM_L1)

    # YCrCb
    img_yuv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YCrCb)

    hist_y = cv2.calcHist([img_yuv], [0], mask, [256], [0, 256])
    hist_cr = cv2.calcHist([img_yuv], [1], mask, [256], [0, 256])
    hist_cb = cv2.calcHist([img_yuv], [2], mask, [256], [0, 256])

    hist_y = cv2.normalize(hist_y, 0, 1, norm_type=cv2.NORM_L1)
    hist_cr = cv2.normalize(hist_cr, 0, 1, norm_type=cv2.NORM_L1)
    hist_cb = cv2.normalize(hist_cb, 0, 1, norm_type=cv2.NORM_L1)

    fig0, ax = plt.subplots(num="Image", figsize=(8, 8))
    ax.imshow(img_rgb)

    fig1, axs = plt.subplots(2, 2, num="Histograms",
                             sharex=False, sharey=False, figsize=(16, 8))

    x = np.arange(256)
    axs[0, 0].plot(x, hist_red, color="red", label="R")
    axs[0, 0].plot(x, hist_green, color="green", label="G")
    axs[0, 0].plot(x, hist_blue, color="blue", label="B")
    axs[0, 0].legend()
    axs[0, 0].set_xlim([0, 256])
    axs[0, 0].set_title("RGB")

    x = np.arange(256)
    axs[0, 1].plot(x, hist_h, color="C0", label="H")
    axs[0, 1].plot(x, hist_s, color="C1", label="S")
    axs[0, 1].plot(x, hist_v, color="C2", label="V")
    axs[0, 1].legend()
    axs[0, 1].set_xlim([0, 256])
    axs[0, 1].set_title("HSV")

    x = np.arange(256)
    axs[1, 0].plot(x, hist_l, color="black", label="L")
    axs[1, 0].plot(x, hist_a, color="magenta", label="a")
    axs[1, 0].plot(x, hist_b, color="yellow", label="b")
    axs[1, 0].legend()
    axs[1, 0].set_xlim([0, 256])
    axs[1, 0].set_title("Lab")

    x = np.arange(256)
    axs[1, 1].plot(x, hist_y, color="black", label="Y")
    axs[1, 1].plot(x, hist_cr, color="red", label="Cr")
    axs[1, 1].plot(x, hist_cb, color="blue", label="Cb")
    axs[1, 1].legend()
    axs[1, 1].set_xlim([0, 256])
    axs[1, 1].set_title("YCrCb")

    plt.show()
