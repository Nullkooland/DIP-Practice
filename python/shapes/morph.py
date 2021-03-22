import cv2
import pyheif
import numpy as np

src_img = pyheif.read_as_numpy("./images/neon_flowers.heic")

cv2.imshow("original", src_img)

gray_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)

# kernel_y = np.array([3, 10, 3], dtype=np.int16)
# kernel_x = np.array([-1, 0, 1], dtype=np.int16)
# grad_img = cv2.sepFilter2D(gray_img, cv2.CV_8U, kernel_x, kernel_y, delta=127)

# cv2.imshow("gradient", grad_img)
# cv2.waitKey()

ret, mask_img = cv2.threshold(gray_img, 100, 255, cv2.THRESH_OTSU)
# mask_img = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 17, 5)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
mask_img = cv2.morphologyEx(mask_img, cv2.MORPH_CLOSE, kernel)
mask_img = cv2.morphologyEx(mask_img, cv2.MORPH_GRADIENT, kernel)

masked_img = cv2.copyTo(src_img, mask_img)
masked_img = cv2.bilateralFilter(masked_img, 7, 3.4, 45)

cv2.imshow("threshold", mask_img)
cv2.imshow("result", masked_img)

cv2.waitKey()