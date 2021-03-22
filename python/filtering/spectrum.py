import cv2
import pyheif
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    img_src = pyheif.read_as_numpy("./images/tiger.heic")
    img_src = cv2.cvtColor(img_src, cv2.COLOR_RGB2GRAY)
    img_src = np.float32(img_src) / 255.0

    spectrum = cv2.dft(img_src, flags=cv2.DFT_COMPLEX_OUTPUT)
    spectrum = np.fft.fftshift(spectrum)
    real, imag = cv2.split(spectrum)

    magnitude = cv2.magnitude(real, imag)
    phase = cv2.phase(real, imag)

    disp_magnitude = cv2.log(magnitude + 1.0)
    disp_magnitude = cv2.normalize(disp_magnitude, None, norm_type=cv2.NORM_MINMAX)

    fig, axs = plt.subplots(1, 3, num="Spectrum", figsize=(18,6))

    axs[0].imshow(img_src, cmap="gray", interpolation="bicubic")
    axs[0].set_title("Original")
    
    axs[1].imshow(disp_magnitude, cmap="inferno", interpolation="bicubic")
    axs[1].set_title("Spectrum Magnitude")

    axs[2].imshow(phase, cmap="hsv", interpolation="nearest")
    axs[2].set_title("Phase")

    plt.show()