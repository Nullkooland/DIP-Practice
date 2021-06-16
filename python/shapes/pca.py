import cv2
from matplotlib import pyplot as plt
import pyheif
import numpy as np

if __name__ == "__main__":
    src_img = pyheif.read_as_numpy("./images/rice.heic")

    # binary threshold to get mask
    ret, mask = cv2.threshold(src_img[..., 0], 25, 255, cv2.THRESH_BINARY)
    # mask = cv2.adaptiveThreshold(src_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 9, 3)

    # apply morphological filters to remove noises
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # get contours
    contours, hierarchy = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    fig, ax = plt.subplots(1, 1, num="Rice", figsize=(12, 8))
    ax.imshow(src_img)

    # compute PCA
    for contour in contours:
        # Draw contours
        contour = np.squeeze(contour, axis=1).astype(np.float32)
        ax.fill(contour[:, 0], contour[:, 1], color="C1", fill=False)

        center, directions, eigenvalues = cv2.PCACompute2(contour, np.empty(0))
        # draw direction vectors for each rice seed
        x, y = center[0]
        v0 = directions[0]
        v1 = directions[1]

        len0 = np.dot(contour - center, v0)
        len1 = np.dot(contour - center, v1)

        v0 *= np.max(len0)
        v1 *= np.max(len1)

        ax.plot(x, y, ls='', marker='o', markersize=8,
                markerfacecolor="None", markeredgecolor="C3")

        ax.arrow(x, y, v0[0], v0[1], color="C0", width=1, head_width=6,
                 head_length=6, length_includes_head=True)
        ax.arrow(x, y, v1[0], v1[1], color="C2", width=0.5, head_width=3,
                 head_length=3, length_includes_head=True)

    plt.show()
