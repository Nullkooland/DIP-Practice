import cv2
import numpy as np
from utils.image_reader import ImageReader
import matplotlib.pyplot as plt

ANGLE_QUANT_LEVEL = 128

if __name__ == "__main__":
    reader = ImageReader()
    src_img = reader.read("images/boss_car_is_ready.heic")
    h, w = src_img.shape[:2]

    edge_map = cv2.Canny(src_img, 40, 150)

    # standard Hough transform
    lines = cv2.HoughLines(edge_map, 1, np.pi / ANGLE_QUANT_LEVEL, 160)
    anno_img_SHT = src_img.copy()

    for line in lines:
        rho, theta = line.flat
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 - h * w * b)
        y1 = int(y0 + h * w * a)
        x2 = int(x0 + h * w * b)
        y2 = int(y0 - h * w * a)
        cv2.line(anno_img_SHT, (x1, y1), (x2, y2), (0, 255, 0), 1, cv2.LINE_AA)

    # probabilistic Hough transform
    lines = cv2.HoughLinesP(edge_map, 1, np.pi / ANGLE_QUANT_LEVEL,
                            60, minLineLength=80, maxLineGap=40)
    anno_img_PHT = src_img.copy()

    for line in lines:
        x1, y1, x2, y2 = line.flat
        cv2.line(anno_img_PHT, (x1, y1), (x2, y2), (0, 255, 0), 1, cv2.LINE_AA)

    plt.figure("Hough Lines", figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.imshow(src_img, cmap="gray")
    plt.title('Ride On: \"Boss, car is ready!\"')

    plt.subplot(2, 2, 2)
    plt.imshow(edge_map, cmap="gray")
    plt.title("Edge")

    plt.subplot(2, 2, 3)
    plt.imshow(anno_img_SHT)
    plt.title("Hough Lines")

    plt.subplot(2, 2, 4)
    plt.imshow(anno_img_PHT)
    plt.title("Probabilistic Hough Lines")

    plt.tight_layout()
    plt.show()
