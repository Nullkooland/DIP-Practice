import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation
import timeit

# Paramaters
N_theta = 256


def radon(img, theta):
    n = theta.shape[0]
    h, w = img.shape[:2]
    d = int(np.ceil(np.sqrt(h ** 2 + w ** 2)))

    p_left = (d - w + 1) // 2
    p_right = (d - w) // 2
    p_top = (d - h + 1) // 2
    p_bottom = (d - h) // 2

    img_padded = cv2.copyMakeBorder(
        img, p_top, p_bottom, p_left, p_right, borderType=cv2.BORDER_CONSTANT, value=0)

    g = np.empty((n, d), dtype=np.float32)

    for i, theta_deg in enumerate(theta):
        rot_mat = cv2.getRotationMatrix2D((d // 2, d // 2), -theta_deg, 1)
        rot_img = cv2.warpAffine(img_padded, rot_mat, (d, d))
        g[i] = np.sum(rot_img, axis=0)

    return g

def get_backprojection_filter(length, ftype='ram-lak'):
    H = np.abs(np.linspace(0, 1, length))
    if ftype == 'shepp-logan':
        H *= np.sinc(np.linspace(0, 1 / 2, length))
    elif ftype == 'hamming':
        H *= 0.54 + 0.46 * np.cos(np.linspace(0, np.pi, length))
    elif ftype == 'cosine':
        H *= np.cos(np.linspace(0, np.pi / 2, length))
    return H


if __name__ == "__main__":
    src_img = cv2.imread('./images/pattern.png')
    src_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
    src_img = np.float32(src_img) / 255.0

    theta = np.linspace(0, 180 - 180 / N_theta, N_theta)
    g = radon(src_img, theta)

    g = np.flip(g)

    d = g.shape[1]
    g /= np.max(g)
    # cv2.imshow('radon', g)

    fig1, axs = plt.subplots(1, 2, num='Radon transform', figsize=(8, 4))
    axs[0].imshow(src_img)
    axs[1].imshow(g, extent=[-d / 2, d / 2, 0, 180],
                  origin='bottom', aspect='auto')
    axs[1].set_xlabel(r'$\alpha$')
    axs[1].set_ylabel(r'$\theta$')

    axs[0].set_title('Original')
    axs[1].set_title('Radon transform')

    fig2, axs = plt.subplots(1, 2, num='Radon reconstruction', figsize=(8, 4))

    rec_img = np.zeros((d, d), dtype=np.float32)

    im_0 = axs[0].imshow(src_img, vmin=-0.1, vmax=0.1)
    im_1 = axs[1].imshow(rec_img)
    
    axs[0].set_title('Smear')
    axs[1].set_title('Reconstructed')

    ims = []

    print('Performing reconstruction...')
    H = get_backprojection_filter(d // 2 + 1, 'shepp-logan')

    for i, theta_deg in enumerate(theta):
        # filter
        g_i = g[i]
        G_i = np.fft.rfft(g_i) * H
        g_i = np.fft.irfft(G_i)

        smear = np.tile(g_i, (d, 1))
        rot_mat = cv2.getRotationMatrix2D((d // 2, d // 2), theta_deg, 1)
        smear = cv2.warpAffine(smear, rot_mat, (d, d))

        rec_img += smear

        im_0 = axs[0].imshow(smear, vmin=-0.1, vmax=0.1, animated=True)
        im_1 = axs[1].imshow(rec_img, animated=True)
        ims.append([im_0, im_1])

    ani = ArtistAnimation(fig2, ims, interval=16, repeat=False, blit=True)
    plt.show()

    rec_img = cv2.normalize(rec_img, None, 0, 1, cv2.NORM_MINMAX)
    cv2.imshow('rec', rec_img)
    cv2.waitKey()
