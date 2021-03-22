import cv2
import pyheif
import numpy as np

src_img = pyheif.read_as_numpy("./images/homography.heic")
h, w, c = src_img.shape

cv2.imshow("Original", src_img)

mask = cv2.inRange(src_img, (0, 0, 90), (80, 80, 255))
morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, morph_kernel)

contours, hierarchy = cv2.findContours(
    mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)

calib_points = np.empty((4, 2), dtype=np.float32)

for i, contour in enumerate(contours):
    center = np.mean(contour, axis=0, dtype=np.float32)
    calib_points[i] = center

cv2.imshow("Red", mask)

dst_points = np.array([
    [1024, 1024],
    [1536, 1024],
    [1024, 512],
    [1536, 512]
], dtype=np.float32)

perspective_mat = cv2.getPerspectiveTransform(calib_points, dst_points)
calib_img = cv2.warpPerspective(src_img, perspective_mat, (w, h), flags=cv2.INTER_CUBIC)

cv2.imshow("Calibrated", calib_img)

gray_img = cv2.cvtColor(calib_img, cv2.COLOR_BGR2GRAY)
gray_img[gray_img == 0] = 255

mask = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 12)
thresh, mask = cv2.threshold(gray_img, 88, 255, cv2.THRESH_BINARY_INV)

morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
mask = cv2.erode(mask, morph_kernel)

morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, morph_kernel, iterations=3)

cv2.imshow("Masked", mask)

cv2.waitKey()
