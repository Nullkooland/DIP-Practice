import cv2
import numpy as np
import matplotlib.pyplot as plt

N = 32
SIGMA = 5.0

if __name__ == "__main__":
    plt.style.use("science")

    n = np.arange(N)
    n_half = n[:N//2]
    x = np.cos(n * 2 * np.pi / N)
    # x = np.exp(-n ** 2 / (2 * SIGMA ** 2))
    # x = np.exp(-(n - N / 2) ** 2 / (SIGMA ** 2))

    x_dft = np.fft.fft(x)
    x_dct = cv2.dct(x[n_half])

    plt.figure("DCT 1D", figsize=(8, 8))

    plt.subplot(3, 1, 1)
    plt.stem(n, x, use_line_collection=True)
    plt.xticks(n)
    plt.title(r"$x[n]$")

    plt.subplot(3, 1, 2)
    plt.stem(n, np.abs(x_dft), use_line_collection=True)
    plt.xticks(n)
    plt.title(r"DFT $|X[k]|$")

    plt.subplot(3, 1, 3)
    plt.stem(n_half, x_dct, use_line_collection=True)
    plt.xticks(n_half)
    plt.title(r"DCT $X[k]$")

    plt.show()