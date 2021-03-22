import cv2
import pyheif
import numpy as np
import matplotlib.pyplot as plt
import pyheif

src_img = pyheif.read_as_numpy("./images/rmb_coins.heic")
h, w = src_img.shape[:2]

gray_img = cv2.cvtColor(src_img, cv2.COLOR_RGB2GRAY)
gray_img = cv2.GaussianBlur(gray_img, (7, 7), 1.5)

circles = cv2.HoughCircles(gray_img, cv2.HOUGH_GRADIENT_ALT,
                           2, 10, param1=280, param2=0.92, minRadius=30, maxRadius=100)

for circle in circles:
    x0, y0, radius = np.uint16(circle).flat
    cv2.drawMarker(src_img, (x0, y0), (0, 225, 255),
                   cv2.MARKER_CROSS, 15, 2, line_type=cv2.LINE_AA)
    cv2.circle(src_img, (x0, y0), radius,
               (255, 0, 255), 3, lineType=cv2.LINE_AA)

plt.imshow(src_img)
plt.title("Hough Circles")
plt.show()
