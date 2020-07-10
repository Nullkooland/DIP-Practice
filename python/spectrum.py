import cv2
import numpy as np
import matplotlib.pyplot as plt

src_img = cv2.imread('./images/tiger.png', cv2.IMREAD_GRAYSCALE)
src_img = np.float32(src_img) / 255.0

plt.figure('Source')
plt.imshow(src_img, cmap='gray', interpolation='bilinear')

spectrum = cv2.dft(src_img, flags=cv2.DFT_COMPLEX_OUTPUT)
spectrum = np.fft.fftshift(spectrum)
real, imag = cv2.split(spectrum)

magnitude = cv2.magnitude(real, imag)
phase = cv2.phase(real, imag)

disp_magnitude = cv2.pow(magnitude, 1.0 / 4.0)
disp_magnitude = cv2.normalize(disp_magnitude, None, norm_type=cv2.NORM_MINMAX)

plt.figure('Spectrum')
plt.imshow(disp_magnitude, cmap='gray', interpolation='bicubic')
plt.show()

disp_magnitude = np.uint8(disp_magnitude * 255.0)
cv2.imwrite('./spectrum_magnitude_lena.png', disp_magnitude)