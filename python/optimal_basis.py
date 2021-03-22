import cv2
import pyheif
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation
from matplotlib.animation import FuncAnimation

PATCH_SIZE = 8
plt.style.use("science")

def img2patches(img, h, w, patch_size):
    patches = np.empty((h * w // (patch_size ** 2),
                        patch_size ** 2), dtype=np.float32)

    for i in range(h // patch_size):
        for j in range(w // patch_size):
            y = i * patch_size
            x = j * patch_size

            patch = img[y:y + patch_size, x:x + patch_size]
            patches[i * w // patch_size + j] = patch.flatten()

    return patches


def patches2img(patches, h, w, patch_size):
    img = np.empty((h, w), dtype=np.float32)

    for k in range(patches.shape[0]):
        y = k // (w // patch_size) * patch_size
        x = k % (w // patch_size) * patch_size

        img[y:y + patch_size, x:x + patch_size] = \
            np.reshape(patches[k], (patch_size, patch_size))

    return img


def snr(a, b):
    return 20 * np.log10(np.mean(np.abs(a)) / np.mean(cv2.absdiff(a, b)))


if __name__ == "__main__":
    src_img = pyheif.read_as_numpy("./images/river2.heic")
    gray_img = cv2.cvtColor(src_img, cv2.COLOR_RGB2GRAY)
    gray_img = np.float32(gray_img) / 255.0
    h, w = gray_img.shape

    # cv2.SVDecomp()

    patches = img2patches(gray_img, h, w, PATCH_SIZE)
    patches_dct = cv2.dct(patches, flags=cv2.DCT_ROWS)
    patches_dft = cv2.dft(patches, flags=cv2.DFT_ROWS | cv2.DFT_COMPLEX_OUTPUT)
    s, u, vt = cv2.SVDecomp(patches)

    sample_range = np.arange(PATCH_SIZE ** 2)

    fig1 = plt.figure(num="SNR", figsize=(8, 4))
    ax_plot = plt.gca()
    ax_plot.set_xlim(0, PATCH_SIZE ** 2)
    ax_plot.set_xlabel("k")
    ax_plot.set_ylabel("SNR (dB)")
    ax_plot.grid()

    snr_dft = np.full(len(sample_range), np.nan)
    snr_dct = np.full(len(sample_range), np.nan)
    snr_svd = np.full(len(sample_range), np.nan)

    snr_dft_line, = ax_plot.plot(sample_range, snr_dft)
    snr_dct_line, = ax_plot.plot(sample_range, snr_dct)
    snr_svd_line, = ax_plot.plot(sample_range, snr_svd)

    snr_dft_line.set_label("DFT")
    snr_dct_line.set_label("DCT")
    snr_svd_line.set_label("SVD")

    ax_plot.legend()

    optimal_basis = patches2img(
        vt, PATCH_SIZE ** 2, PATCH_SIZE ** 2, PATCH_SIZE)

    # plt.figure("Optimal basis")
    # plt.imshow(optimal_basis)

    # for i in range(1, PATCH_SIZE):
    #     plt.plot([0, PATCH_SIZE ** 2],
    #              [i * PATCH_SIZE, i * PATCH_SIZE], color="red")
    #     plt.plot([i * PATCH_SIZE, i * PATCH_SIZE],
    #              [0, PATCH_SIZE ** 2], color="red")

    # plt.axis([0, PATCH_SIZE ** 2, 0, PATCH_SIZE ** 2])
    # plt.show()

    # patches_dct_partial = np.empty_like(patches_dct)
    # patches_dft_partial = np.empty_like(patches_dft)
    # u_partial = np.empty_like(u)

    fig2, axs = plt.subplots(1, 3, num="Reconstruction", figsize=(12, 4))

    axs[0].set_title("DFT")
    axs[1].set_title("DCT")
    axs[2].set_title("SVD")

    im_dft = axs[0].imshow(gray_img, animated=True)
    im_dct = axs[1].imshow(gray_img, animated=True)
    im_svd = axs[2].imshow(gray_img, animated=True)

    def update_rec_img(i):
        patches_dft_partial = np.copy(patches_dft)
        patches_dct_partial = np.copy(patches_dct)
        s_partial = np.copy(s)

        patches_dft_partial[:, i:, :] = 0
        patches_dct_partial[:, i:] = 0
        s_partial[i:] = 0

        patches_dft_rec_c = cv2.idft(
            patches_dft_partial, flags=cv2.DFT_ROWS | cv2.DFT_COMPLEX_OUTPUT | cv2.DFT_SCALE)
        patches_dft_rec = patches_dft_rec_c[..., 0]
        patches_dct_rec = cv2.idct(patches_dct_partial, flags=cv2.DCT_ROWS)
        patches_svd_rec = u @ (s_partial * vt)

        img_dft_rec = patches2img(patches_dft_rec, h, w, PATCH_SIZE)
        img_dct_rec = patches2img(patches_dct_rec, h, w, PATCH_SIZE)
        img_svd_rec = patches2img(patches_svd_rec, h, w, PATCH_SIZE)

        im_dft.set_data(img_dft_rec)
        im_dct.set_data(img_dct_rec)
        im_svd.set_data(img_svd_rec)
        fig1.canvas.draw_idle()

        snr_dft[i] = snr(gray_img, img_dft_rec)
        snr_dct[i] = snr(gray_img, img_dct_rec)
        snr_svd[i] = snr(gray_img, img_svd_rec)

        snr_dft_line.set_data(sample_range, snr_dft)
        snr_dct_line.set_data(sample_range, snr_dct)
        snr_svd_line.set_data(sample_range, snr_svd)
        ax_plot.set_ylim(0, 1.25 * max(snr_dft[i], snr_dct[i], snr_svd[i]))
        fig2.canvas.draw_idle()

        return [im_dft, im_dct, im_svd]

    # ani_ims = ArtistAnimation(fig1, ims, interval=100, repeat=False, blit=True)
    ani_ims = FuncAnimation(
        fig2, update_rec_img, sample_range, interval=100, repeat=False, blit=True)

    plt.show()

    # extract and vectorize patches
    print(patches.shape)
