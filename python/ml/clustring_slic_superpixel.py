import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils.image_reader import ImageReader

if __name__ == "__main__":
    reader = ImageReader()
    img_src = reader.read("images/dog.heic")
    # h, w = img_src.shape[:2]

    slic = cv2.ximgproc.createSuperpixelSLIC(
        img_src, cv2.ximgproc.SLICO, 32, 90)
    # slic.enforceLabelConnectivity()
    slic.iterate(5)

    contours_mask = slic.getLabelContourMask()
    num_superpixels = slic.getNumberOfSuperpixels()
    labels = slic.getLabels()
    labels_color = np.empty_like(img_src)
    
    mean_colors = np.empty((num_superpixels, 3), dtype=np.uint8)
    for i in range(num_superpixels):
        mask = (labels == i)
        mask_cv = np.uint8(mask)
        mean_color = cv2.mean(img_src, mask=mask_cv)
        mean_colors[i] = np.uint8(mean_color[:3])

    labels_color = mean_colors[labels]

    fig, axs = plt.subplots(1, 2, num="SLIC", figsize=(14, 6))

    # cv2.addWeighted(img_src, 0.25, labels_color, 0.75, 0, dst=img_src)

    contours_mask_np = np.bool8(contours_mask)
    img_src[contours_mask_np] = (255, 0, 100)

    # cv2.medianBlur(labels_color, 5, dst=labels_color)

    axs[0].imshow(img_src)
    axs[1].imshow(labels_color)
    axs[0].set_title("Original")
    axs[1].set_title("Super pixels")
    plt.show()
