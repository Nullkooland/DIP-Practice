import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils.image_reader import ImageReader

PIXEL_SIZE = 4
PIXEL_MARGIN = 2

if __name__ == "__main__":
    reader = ImageReader()
    img_src = reader.read("images/testcard.heic")
    img_src = cv2.cvtColor(img_src, cv2.COLOR_BGR2RGB)

    h, w = img_src.shape[:2]

    img_stylized = np.zeros_like(img_src)
    mask = np.zeros((h, w), dtype=np.uint8)

    # form pixel-like style image
    for i in range(0, h - PIXEL_MARGIN, PIXEL_SIZE + PIXEL_MARGIN):
        for j in range(0, w - PIXEL_MARGIN, PIXEL_SIZE + PIXEL_MARGIN):
            mask[i + PIXEL_MARGIN:i + PIXEL_MARGIN +
                 PIXEL_SIZE, j + PIXEL_MARGIN: j + PIXEL_MARGIN + PIXEL_SIZE] = 255

            roi_src = img_src[i + PIXEL_MARGIN:i + PIXEL_MARGIN +
                              PIXEL_SIZE, j + PIXEL_MARGIN: j + PIXEL_MARGIN + PIXEL_SIZE]

            roi_dst = img_stylized[i + PIXEL_MARGIN:i + PIXEL_MARGIN +
                                   PIXEL_SIZE, j + PIXEL_MARGIN:j + PIXEL_MARGIN + PIXEL_SIZE]

            roi_dst[:] = cv2.mean(roi_src)[:3]

    # add glow effects
    layer_glow = cv2.GaussianBlur(img_stylized, (9, 9), 1.5)
    img_stylized += cv2.add(img_stylized, layer_glow, mask=~mask)

    fig, (ax_src, ax_stylized) = plt.subplots(
        1, 2, num="Pixel Stylized", figsize=(12, 6))
    ax_src.imshow(img_src)
    ax_src.set_title("Original")

    ax_stylized.imshow(img_stylized)
    ax_stylized.set_title("Stylized")

    plt.show()
