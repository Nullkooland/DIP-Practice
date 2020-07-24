import numpy as np
import cv2
import matplotlib.pyplot as plt

plt.style.use('science')

STD_NOISE = 0.25
NUM_IMG = 3
NUM_HIST_BIN = 2048
RANGE_HIST = (-5, 5)

if __name__ == "__main__":
    src_img = cv2.imread('./images/city.png', cv2.IMREAD_GRAYSCALE)
    src_img = np.float32(src_img) / 255.0
    M, N = src_img.shape

    blurred_img = cv2.GaussianBlur(src_img, (9, 9), 0)
    noised_img = src_img + np.random.randn(M, N).astype(np.float32) * STD_NOISE

    imgs = np.stack((src_img, blurred_img, noised_img))
    gradient_hists = np.empty((NUM_IMG, NUM_HIST_BIN))

    hist_channels = np.zeros(NUM_IMG, dtype=int)

    for i, img in enumerate(imgs):
        g_x = cv2.Sobel(img, cv2.CV_32F, 1, 0)
        g_y = cv2.Sobel(img, cv2.CV_32F, 0, 1)
        g_norm = cv2.magnitude(g_x, g_y) * np.sign(g_y)
        gradient_hists[i] = np.histogram(g_norm, NUM_HIST_BIN, RANGE_HIST)[0]
        
    r = np.linspace(RANGE_HIST[0], RANGE_HIST[1], NUM_HIST_BIN)
    plt.figure('Gradient distribution', figsize=(12, 4))
    for i in range(NUM_IMG):
        log_gradient_hist = np.log10(gradient_hists[i] + 1)
        plt.plot(r, log_gradient_hist)

    plt.legend(['Original', 'Blurred', 'Noised'])

    plt.show()