import numpy as np
import cv2
import matplotlib.pyplot as plt
from utils.image_reader import ImageReader


STD_NOISE = 0.25
NUM_IMG = 3
NUM_HIST_BIN = 2048
RANGE_HIST = (-5, 5)

if __name__ == "__main__":
    plt.style.use("science")

    reader = ImageReader()
    img_src = reader.read("images/city.heic")
    img_src = cv2.cvtColor(img_src, cv2.COLOR_RGB2GRAY)
    img_src = np.float32(img_src) / 255.0
    M, N = img_src.shape

    img_blurred = cv2.GaussianBlur(img_src, (9, 9), 0)
    img_noised = img_src + np.random.randn(M, N).astype(np.float32) * STD_NOISE

    imgs = np.stack((img_src, img_blurred, img_noised))
    gradient_hists = np.empty((NUM_IMG, NUM_HIST_BIN))

    hist_channels = np.zeros(NUM_IMG, dtype=int)

    for i, img in enumerate(imgs):
        g_x = cv2.Sobel(img, cv2.CV_32F, 1, 0)
        g_y = cv2.Sobel(img, cv2.CV_32F, 0, 1)
        g_norm = cv2.magnitude(g_x, g_y) * np.sign(g_y)
        gradient_hists[i] = np.histogram(g_norm, NUM_HIST_BIN, RANGE_HIST)[0]

    r = np.linspace(RANGE_HIST[0], RANGE_HIST[1], NUM_HIST_BIN)
    plt.figure("Gradient distribution", figsize=(12, 4))
    for i in range(NUM_IMG):
        log_gradient_hist = np.log10(gradient_hists[i] + 1)
        plt.plot(r, log_gradient_hist)

    plt.legend(["Original", "Blurred", "Noised"])

    plt.show()
