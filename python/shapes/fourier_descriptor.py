import cv2
import pyheif
import matplotlib.pyplot as plt

DIM_FOURIER_DESC = 256
LEN_HPF = DIM_FOURIER_DESC // 2

if __name__ == "__main__":
    img_src = pyheif.read_as_numpy("images/hand.heic")
    mask = img_src[..., 3]

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    fig, axs = plt.subplots(1, 2, num="Fourier descriptors", figsize=(10, 6))
    axs[0].imshow(mask)
    axs[0].set_title("Mask")

    for contour in contours:
        if len(contour) < 1000:
            continue

        contour = cv2.ximgproc.contourSampling(contour, DIM_FOURIER_DESC)
        fourier_desc = cv2.dft(contour, flags=cv2.DFT_COMPLEX_INPUT)

        # Filter out high frequency harmonics
        fourier_desc[(DIM_FOURIER_DESC - LEN_HPF) //
                     2: (DIM_FOURIER_DESC + LEN_HPF) // 2, ...] = 0

        contour_rec = cv2.idft(
            fourier_desc, flags=cv2.DFT_COMPLEX_OUTPUT | cv2.DFT_SCALE)

        axs[1].plot(contour[..., 0], contour[..., 1],
                    alpha=0.5, label="Original")
        axs[1].plot(contour_rec[..., 0], contour_rec[..., 1],
                    alpha=0.75, label="Approximate")
        axs[1].axis("equal")
        axs[1].set_title("Contour")
        axs[1].legend(loc="upper right")

    plt.show()
    print("Done")
