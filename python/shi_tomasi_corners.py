import cv2
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # load image
    img_src = cv2.imread('./images/chessboard_calib.png', cv2.IMREAD_UNCHANGED)
    alpha = img_src[..., 3]
    img_src = np.copy(img_src[..., :3])
    img_src[alpha == 0] = (0, 200, 0)

    img_2x = cv2.resize(img_src, None, fx=2.0, fy=2.0,
                        interpolation=cv2.INTER_CUBIC)
    img_gray = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)

    # locate corners using Shi-Tomasi algorithm
    corners = cv2.goodFeaturesToTrack(img_gray, 200, 0.1, 20, k=5)

    for corner in corners:
        pos = tuple(np.int32(corner.flatten()))
        cv2.circle(img_src, pos, 3, (255, 0, 0), -1, cv2.LINE_AA)

    # Refine to get sub-pixel corners
    criteria = (cv2.TermCriteria_COUNT + cv2.TermCriteria_EPS, 32, 1e-2)
    corners = cv2.cornerSubPix(img_gray, corners, (5, 5), (-1, -1), criteria)

    for corner in corners:
        pos = tuple(np.int32(corner.flatten() * 2.0))
        cv2.circle(img_2x, pos, 4, (255, 0, 255), -1, cv2.LINE_AA)

    fig, (ax_denoised, ax_corners, ax_corners_refined) = plt.subplots(
        1, 3, num='Shi-Tomasi Corners', figsize=(12, 4))

    ax_denoised.imshow(img_gray)
    ax_denoised.set_title('Source')

    ax_corners.imshow(img_src)
    ax_corners.set_title('Corners')

    ax_corners_refined.imshow(img_2x)
    ax_corners_refined.set_title('Refined Corners')

    plt.show()
