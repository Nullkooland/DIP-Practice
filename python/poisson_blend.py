import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils.image_reader import ImageReader

if __name__ == "__main__":
    # Load images
    reader = ImageReader()
    foreground = reader.read("images/polar_bear.heic")
    background = reader.read("images/loquat_painting.heic")

    # foreground = cv2.resize(foreground, None, fx=0.5, fy=0.5)
    mask = np.full_like(foreground, 255)

    (height, width, channels) = background.shape
    pos = (120, 64)

    blended = cv2.seamlessClone(foreground, background, mask, pos, cv2.MIXED_CLONE)

    fig, axs = plt.subplots(1, 3, num="Possion blend", figsize=(12, 4))

    axs[0].imshow(background)
    axs[0].set_title("Background")

    axs[1].imshow(foreground)
    axs[1].set_title("Foreground")

    axs[2].imshow(blended)
    axs[2].set_title("Blended")

    plt.show()
