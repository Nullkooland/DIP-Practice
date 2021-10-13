import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils.image_reader import ImageReader

if __name__ == "__main__":
    reader = ImageReader()
    src_img = reader.read("images/boat.heic")
    # psf_img = reader.read("images/blur_impulse.heic")

    f = src_img.astype(np.float32) / 255.0

    # h = np.ones((9, 9)) / 81
    h = np.eye(17) / 17

    g = cv2.filter2D(f, cv2.CV_32F, h)
    g_n = g + np.random.randn(*f.shape) * 0.01

    def richardson_lucy(g, h, iterations):
        f_est = np.full(g.shape, 0.5)  # initial estimation
        h_flip = np.flip(h)  # flipped psf

        for _ in range(iterations):
            g_est = cv2.filter2D(f_est, cv2.CV_32F, h)
            err_est = cv2.filter2D(g / g_est, cv2.CV_32F, h_flip)
            f_est *= err_est

        return f_est

    f_r_10 = richardson_lucy(g_n, h, 10)
    f_r_50 = richardson_lucy(g_n, h, 50)
    f_r_100 = richardson_lucy(g_n, h, 100)

    fig, axs = plt.subplots(2, 3, num="Richardson-Lucy", figsize=(16, 8))

    axs[0, 0].imshow(f, cmap="gray", vmin=0, vmax=1)
    axs[0, 0].set_title("Original")

    axs[0, 1].imshow(g, cmap="gray", vmin=0, vmax=1)
    axs[0, 1].set_title("Blur")

    axs[0, 2].imshow(g_n, cmap="gray", vmin=0, vmax=1)
    axs[0, 2].set_title("Blur (noise)")

    axs[1, 0].imshow(f_r_10, cmap="gray", vmin=0, vmax=1)
    axs[1, 0].set_title("Recovered: iterations=10")

    axs[1, 1].imshow(f_r_50, cmap="gray", vmin=0, vmax=1)
    axs[1, 1].set_title("Recovered: iterations=30")

    axs[1, 2].imshow(f_r_100, cmap="gray", vmin=0, vmax=1)
    axs[1, 2].set_title("Recovered: iterations=80")

    plt.show()
