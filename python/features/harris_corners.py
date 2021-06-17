import cv2
import pyheif
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    img_src = pyheif.read_as_numpy("./images/loquat_painting.heic")
    img_gray = cv2.cvtColor(img_src, cv2.COLOR_RGB2GRAY)

    corners_response = cv2.cornerHarris(img_gray, 4, 3, 0.06)

    thresh = np.max(corners_response) * 0.02
    thresh, corners_locs = cv2.threshold(
        corners_response, thresh, 255, cv2.THRESH_BINARY)

    corners_locs = np.uint8(corners_locs)

    num_corners, _, _, corners = cv2.connectedComponentsWithStats(
        corners_locs)

    # Refine to achieve sub-pixel precision
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners_refined = cv2.cornerSubPix(img_gray, np.float32(
        corners), (5, 5), (-1, -1), criteria)

    # Show corners
    fig, ax = plt.subplots(1, 1, num="Harris corners", figsize=(8, 8))
    ax.imshow(img_src)
    for corner, corner_refined in zip(corners, corners_refined):
        ax.plot(corner[0], corner[1], ls='', marker='.', color="red")
        ax.plot(corner_refined[0], corner_refined[1], ls='', marker='.', color="magenta")

    plt.show()
