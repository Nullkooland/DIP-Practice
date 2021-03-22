import cv2
import pyheif
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    img_src = pyheif.read_as_numpy("./images/opossum.heic")

    # saliency = cv2.saliency.StaticSaliencyFineGrained_create()
    saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
    ret, saliency_map = saliency.computeSaliency(img_src)

    fig, (ax_src, ax_saliency) = plt.subplots(
        1, 2, num="Static Saliency", figsize=(8, 4))

    ax_src.imshow(img_src)
    ax_src.set_title("Original")

    ax_saliency.imshow(saliency_map, cmap="viridis")
    ax_saliency.set_title("Saliency Map")

    plt.show()
