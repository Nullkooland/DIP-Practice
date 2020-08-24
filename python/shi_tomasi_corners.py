import cv2
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # load image
    img_src = cv2.imread('./images/pattern.png')
    img_gray = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)

    # remove pepper noise in the src image
    kernel_0 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
    kernel_1 = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 2))
    img_gray = cv2.morphologyEx(img_gray, cv2.MORPH_CLOSE, kernel_0)
    img_gray = cv2.morphologyEx(img_gray, cv2.MORPH_CLOSE, kernel_1)

    # locate corners using Shi-Tomasi algorithm
    corners = cv2.goodFeaturesToTrack(img_gray, 128, 0.6, 5)

    for corner in corners:
        pos = tuple(np.int32(corner.flatten()))
        cv2.circle(img_src, pos, 3, (255, 0, 0), -1, cv2.LINE_AA)

    fig, (ax_denoised, ax_corners) = plt.subplots(
        1, 2, num='Shi-Tomasi Corners', figsize=(12, 6))

    ax_denoised.imshow(img_gray)
    ax_denoised.set_title('Denoised')

    ax_corners.imshow(img_src)
    ax_corners.set_title('Corners')

    plt.show()
