import cv2
import numpy as np
from utils.image_reader import ImageReader
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # load image
    reader = ImageReader()
    img_src = reader.read("images/chessboard_calib.heic", ignore_alpha=False)
    alpha = img_src[..., 3]
    img_src = np.copy(img_src[..., :3])
    img_src[alpha == 0] = (0, 200, 0)

    img_2x = cv2.resize(img_src, None, fx=2.0, fy=2.0,
                        interpolation=cv2.INTER_CUBIC)
    img_gray = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)

    # locate corners using Shi-Tomasi algorithm
    corners = cv2.goodFeaturesToTrack(img_gray, 200, 0.1, 20, k=5).reshape(-1, 2)

    # Refine to get sub-pixel corners
    criteria = (cv2.TermCriteria_COUNT + cv2.TermCriteria_EPS, 32, 1e-2)
    corners_refined = cv2.cornerSubPix(img_gray, corners, (5, 5), (-1, -1), criteria)

    # Show corners
    fig, ax = plt.subplots(1, 1, num="Shi-Tomasi corners", figsize=(8, 8))
    ax.imshow(img_src)

    for corner, corners_refined in zip(corners, corners_refined):
        ax.plot(corner[0], corner[1], ls='', marker='.', color="red")
        ax.plot(corners_refined[0], corners_refined[1], ls='', marker='.', color="magenta")

    plt.show()
