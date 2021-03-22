import cv2
import pyheif
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

BLUR_KERNEL_SIZE = 13
NOISE_STD = 0.0025

src_img = pyheif.read_as_numpy("./images/river.heic")
height, width= src_img.shape[:2]
# psf_img = pyheif.read_as_numpy("./images/blur_impulse.heic")

f = cv2.cvtColor(src_img, cv2.COLOR_RGB2GRAY).astype(np.float64) / 255.0
# h = cv2.cvtColor(psf_img, cv2.COLOR_RGB2GRAY).astype(np.float64) / 255.0

h = cv2.getGaussianKernel(BLUR_KERNEL_SIZE, 0)
h = h @ h.T

# normalize psf
h /= np.sum(h)

# zero-padding h to make spectrum size matched
pad_v = height // 2 - BLUR_KERNEL_SIZE // 2
pad_h = width // 2 - BLUR_KERNEL_SIZE // 2
h = cv2.copyMakeBorder(h, pad_v - 1, pad_v, pad_h - 1, pad_h, cv2.BORDER_CONSTANT)
h = np.fft.fftshift(h)

# cv2.imshow("wahla", h / np.max(h))
# cv2.waitKey()

# additive gaussian white noise
n = np.random.randn(*f.shape) * NOISE_STD

# calculate spectrum of g, h, n
F = np.fft.fft2(f)
H = np.fft.fft2(h)
N = np.fft.fft2(n)

# image distorted by psf and noise
G = H * F + N

# calculate power spectra of g, h, n
P_g = np.abs(G) ** 2
P_h = np.abs(H) ** 2
P_n = np.abs(N) ** 2

# wiener filtering
SNR = P_g / P_n
F_r = G * np.conjugate(H) / (P_h + 1.0 / SNR)

# retrieve images in spatial domain
g = np.fft.ifft2(G)
g = np.real(g)

f_r = np.fft.ifft2(F_r)
f_r = np.real(f_r)

plt.figure("Wiener Filter", figsize=(12, 5))

plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(f, cmap="gray", vmin=0, vmax=1)
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("Deteriorated Image")
plt.imshow(g, cmap="gray", vmin=0, vmax=1)
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title("Restored Image")
plt.imshow(f_r, cmap="gray", vmin=0, vmax=1)
plt.axis("off")

plt.tight_layout()
plt.show()
