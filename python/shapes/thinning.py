import cv2
import pyheif
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

if __name__ == "__main__":
    img_src = pyheif.read_as_numpy("images/hand.heic")
    mask = img_src[..., 3]
    skeleton = cv2.ximgproc.thinning(
        mask, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)

    dist = cv2.distanceTransform(mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    dist_laplacian = cv2.Laplacian(dist, cv2.CV_32F, ksize=5)

    fig, axs = plt.subplots(1, 3, num="Thinning", figsize=(14, 6))
    axs[0].imshow(mask)
    axs[0].set_title("Mask")

    axs[1].imshow(skeleton)
    axs[1].set_title("Thinging Skeleton")

    im = axs[2].imshow(dist_laplacian, cmap="Spectral")
    axs[2].set_title("Distance transform + Laplacian")

    divider = make_axes_locatable(axs[2])
    cax = divider.append_axes("right", size="5%", pad="2%")
    fig.colorbar(im, cax=cax)

    plt.show()
