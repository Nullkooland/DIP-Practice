import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils.image_reader import ImageReader

if __name__ == "__main__":
    reader = ImageReader()
    src_img = reader.read("images/corn_kernels.heic", ignore_alpha=False)
    mask = src_img[..., 3]

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    cv2.imshow("corns", src_img)
    cv2.imshow("mask", mask)

    mask = ~mask

    # mask = np.ones((512, 512), dtype=np.uint8)

    # mask[0, 511] = 0
    # mask[256, 256] = 0
    # mask[128, 128] = 0
    # mask[480, 80] = 0

    dist_l2 = cv2.distanceTransform(mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    dist_l1 = cv2.distanceTransform(mask, cv2.DIST_L1, cv2.DIST_MASK_3)
    dist_linf = cv2.distanceTransform(mask, cv2.DIST_C, cv2.DIST_MASK_5)

    fig, axs = plt.subplots(1, 4, num="Distance Transform", figsize=(16, 4))

    axs[0].imshow(~mask)
    axs[0].set_title("Mask")

    axs[1].imshow(dist_l2, cmap="inferno")
    axs[1].set_title(r"$\ell_2$" + " (Euclidean Distance)")

    axs[2].imshow(dist_l1, cmap="inferno")
    axs[2].set_title(r"$\ell_1$" + " (City Distance)")

    axs[3].imshow(dist_linf, cmap="inferno")
    axs[3].set_title(r"$\ell_\infty$" + " (Checkerboard Distance)")

    plt.tight_layout()
    plt.show()