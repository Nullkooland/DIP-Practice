import cv2
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    img_src = cv2.imread('./images/quiet_street.png')
    img_src = cv2.cvtColor(img_src, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img_src, cv2.COLOR_RGB2GRAY)

    corners_response = cv2.cornerHarris(img_gray, 4, 3, 0.05)

    thresh = np.max(corners_response) * 0.01
    thresh, corners_locs = cv2.threshold(
        corners_response, thresh, 255, cv2.THRESH_BINARY)

    corners_locs = np.uint8(corners_locs)

    num_corners, _, _, centroids = cv2.connectedComponentsWithStats(
        corners_locs)

    # cv2.cornerSubPix()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv2.cornerSubPix(img_gray, np.float32(
        centroids), (5, 5), (-1, -1), criteria)

    for i in range(0, num_corners):
        pos = tuple(np.int32(corners[i]))
        cv2.drawMarker(img_src, pos, (0, 0, 255),
                       cv2.MARKER_TILTED_CROSS, 9, 1, cv2.LINE_AA)

    plt.figure('Harris Corners', figsize=(12, 6))
    plt.imshow(img_src)
    plt.show()
