import cv2
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np


def noop(value):
    pass


if __name__ == "__main__":
    cv2.namedWindow('src')
    cv2.namedWindow('gaussian blur')
    cv2.namedWindow('median blur')

    src_img = cv2.imread('./images/tiger.png')

    # (b, g, r) = cv2.split(src_img)
    # r = cv2.equalizeHist(r)
    # g = cv2.equalizeHist(g)
    # b = cv2.equalizeHist(b)
    # src_img = cv2.merge((b, g, r))

    cv2.imshow('src', src_img)

    cv2.createTrackbar('kernel size', 'gaussian blur', 3, 37, noop)
    cv2.createTrackbar('sigma X', 'gaussian blur', 0, 100, noop)
    cv2.createTrackbar('sigma Y', 'gaussian blur', 0, 100, noop)

    cv2.createTrackbar('kernel size', 'median blur', 3, 37, noop)

    while True:
        ksize = cv2.getTrackbarPos('kernel size', 'gaussian blur')
        sigma_x = cv2.getTrackbarPos('sigma X', 'gaussian blur') * 0.1
        sigma_y = cv2.getTrackbarPos('sigma Y', 'gaussian blur') * 0.1

        if ksize % 2 == 0:
            ksize += 1

        blur_img = cv2.GaussianBlur(src_img, (ksize, ksize),
                                    sigmaX=sigma_x,
                                    sigmaY=sigma_y)
        cv2.imshow('gaussian blur', blur_img)

        ksize = cv2.getTrackbarPos('kernel size', 'median blur')

        if ksize % 2 == 0:
            ksize += 1

        blur_img = cv2.medianBlur(src_img, ksize)
        cv2.imshow('median blur', blur_img)

        if cv2.waitKey(33) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
