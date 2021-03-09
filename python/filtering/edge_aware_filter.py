import cv2
import numpy as np
import pyheif
import matplotlib.pyplot as plt

if __name__ == "__main__":
    img_src = pyheif.read_as_numpy('./images/trek.heic')

    img_sketch, img_sketch = cv2.pencilSketch(
        img_src, None, None, 50, 0.08, 0.05)
    img_skin_smoothed = cv2.edgePreservingFilter(
        img_src, None, cv2.RECURS_FILTER, 50, 0.1)
    img_detail_enhanced = cv2.detailEnhance(img_src, None, 50, 0.06)

    fig, axs = plt.subplots(2, 2, num='Edge aware filtering', figsize=(8, 8))
    axs[0, 0].imshow(img_src)
    axs[0, 1].imshow(img_sketch)
    axs[1, 0].imshow(img_skin_smoothed)
    axs[1, 1].imshow(img_detail_enhanced)

    axs[0, 0].set_title('Original')
    axs[0, 1].set_title('Pencil sketch')
    axs[1, 0].set_title('Edge-preserving smooth')
    axs[1, 1].set_title('Detail enhance')

    plt.show()
