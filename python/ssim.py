import cv2
import pyheif
import numpy as np
import matplotlib.pyplot as plt


def get_psnr(img_x, img_y):
    h, w, c = img_x.shape
    diff = img_x - img_y
    mse = np.sum(diff ** 2) / (w * h * c)
    return 20 * np.log10(255) - 10 * np.log10(mse)


def get_ssim(img_x, img_y):
    c1 = 6.5025  # assume L = 255
    c2 = 58.5225

    img_x = np.float32(img_x)
    img_y = np.float32(img_y)

    mean_x = cv2.GaussianBlur(img_x, (11, 11), 1.5)  # E[X]
    mean_y = cv2.GaussianBlur(img_y, (11, 11), 1.5)  # E[Y]

    mean_x_square = cv2.GaussianBlur(img_x ** 2, (11, 11), 1.5)  # E[X^2]
    mean_y_square = cv2.GaussianBlur(img_y ** 2, (11, 11), 1.5)  # E[Y^2]
    mean_xy = cv2.GaussianBlur(img_x * img_y, (11, 11), 1.5)  # E[XY]

    variance_x = mean_x_square - mean_x ** 2  # D[X] = E[X^2] - E[X]^2
    variance_y = mean_y_square - mean_y ** 2  # D[Y] = E[Y^2] - E[Y]^2
    covariance_xy = mean_xy - mean_x * mean_y  # cov(X, Y) = E[XY] - E[X]E[Y]

    ssim = \
        ((2 * mean_x * mean_y + c1) * (2 * covariance_xy + c2)) / \
        ((mean_x ** 2 + mean_y ** 2 + c1) * (variance_x + variance_y + c2))

    h, w, c = ssim.shape
    mssim = np.mean(ssim)

    return mssim, ssim


def get_ms_ssim(img_x, img_y, n_level=5):
    c1 = 6.5025  # assume L = 255
    c2 = 58.5225
    weights_scaled = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]

    cs = 1.0

    img_x = np.float32(img_x)
    img_y = np.float32(img_y)

    for i in range(n_level):
        mean_x = cv2.GaussianBlur(img_x, (11, 11), 1.5)  # E[X]
        mean_y = cv2.GaussianBlur(img_y, (11, 11), 1.5)  # E[Y]

        mean_x_square = cv2.GaussianBlur(img_x ** 2, (11, 11), 1.5)  # E[X^2]
        mean_y_square = cv2.GaussianBlur(img_y ** 2, (11, 11), 1.5)  # E[Y^2]
        mean_xy = cv2.GaussianBlur(img_x * img_y, (11, 11), 1.5)  # E[XY]

        variance_x = mean_x_square - mean_x ** 2  # D[X] = E[X^2] - E[X]^2
        variance_y = mean_y_square - mean_y ** 2  # D[Y] = E[Y^2] - E[Y]^2
        covariance_xy = mean_xy - mean_x * \
            mean_y  # cov(X, Y) = E[XY] - E[X]E[Y]

        cs_map = (2 * covariance_xy + c2) / (variance_x + variance_y + c2)
        cs *= np.mean(cs_map) ** weights_scaled[i]

        # img_x = cv2.pyrDown(img_x)
        # img_y = cv2.pyrDown(img_y)
        img_x = cv2.resize(img_x, None, fx=0.5, fy=0.5,
                           interpolation=cv2.INTER_AREA)
        img_y = cv2.resize(img_y, None, fx=0.5, fy=0.5,
                           interpolation=cv2.INTER_AREA)

    l_map = (2 * mean_x * mean_y + c1) / (mean_x ** 2 + mean_y ** 2 + c1)
    return np.mean(l_map) * cs


if __name__ == "__main__":
    src_img = pyheif.read_as_numpy("./images/tiger.heic")
    h, w, c = src_img.shape

    small_img = cv2.resize(src_img, None, fx=0.25, fy=0.25,
                           interpolation=cv2.INTER_AREA)

    inter_img_nearest = cv2.resize(
        small_img, (w, h), interpolation=cv2.INTER_NEAREST)

    inter_img_linear = cv2.resize(
        small_img, (w, h), interpolation=cv2.INTER_LINEAR)

    inter_img_cubic = cv2.resize(
        small_img, (w, h), interpolation=cv2.INTER_CUBIC)

    inter_img_lanczos4 = cv2.resize(
        small_img, (w, h), interpolation=cv2.INTER_LANCZOS4)

    # inter_img_cubic = cv2.GaussianBlur(inter_img_cubic, (17,17), 0)

    psnr_nearest = get_psnr(src_img, inter_img_nearest)
    psnr_linear = get_psnr(src_img, inter_img_linear)
    psnr_cubic = get_psnr(src_img, inter_img_cubic)
    psnr_lanczos4 = get_psnr(src_img, inter_img_lanczos4)

    mssim_src, ssim_src = get_ssim(src_img, src_img)
    mssim_nearest, ssim_nearest = get_ssim(src_img, inter_img_nearest)
    mssim_linear, ssim_linear = get_ssim(src_img, inter_img_linear)
    mssim_cubic, ssim_cubic = get_ssim(src_img, inter_img_cubic)
    mssim_lanczos4, ssim_lanczos4 = get_ssim(src_img, inter_img_lanczos4)

    # blured_img = cv2.blur(src_img, (23, 23))
    # rua_0 = get_psnr(src_img, blured_img)
    # rua_1, _ = get_ssim(src_img, blured_img)
    # rua_2 = get_ms_ssim(src_img, blured_img)

    # ms_ssim_src = get_ms_ssim(src_img, src_img)
    # ms_ssim_nearest = get_ms_ssim(src_img, inter_img_nearest)
    # ms_ssim_linear = get_ms_ssim(src_img, inter_img_linear)
    # ms_ssim_cubic = get_ms_ssim(src_img, inter_img_cubic)
    # ms_ssim_lanczos4 = get_ms_ssim(src_img, inter_img_lanczos4)

    fig, axs = plt.subplots(2, 5, num="Upsampling loss comparision", figsize=(16, 7.5))

    axs[0, 0].imshow(src_img)
    axs[0, 0].title("Original")

    axs[1, 0].imshow(ssim_src, vmin=0, vmax=1)
    axs[1, 0].title("SSIM - Original")
    axs[1, 0].xlabel(f"\nMSSIM = {mssim_src:.4f}")

    axs[0, 0].subplot(2, 5, 2)
    axs[0, 0].imshow(inter_img_nearest)
    axs[0, 0].title("Nearest")
    axs[0, 0].xlabel(f"PSNR = {psnr_nearest:.2f} dB")

    axs[0, 0].subplot(2, 5, 7)
    axs[0, 0].imshow(ssim_nearest, vmin=0, vmax=1)
    axs[0, 0].title("SSIM - Nearest")
    axs[0, 0].xlabel(f"\nMSSIM = {mssim_nearest:.4f}")

    axs[0, 0].subplot(2, 5, 3)
    axs[0, 0].imshow(inter_img_linear)
    axs[0, 0].title("Bilinear")
    axs[0, 0].xlabel(f"PSNR = {psnr_linear:.2f} dB")

    axs[0, 0].subplot(2, 5, 8)
    axs[0, 0].imshow(ssim_linear, vmin=0, vmax=1)
    axs[0, 0].title("SSIM - Bilinear")
    axs[0, 0].xlabel(f"\nMSSIM = {mssim_linear:.4f}")

    axs[0, 0].subplot(2, 5, 4)
    axs[0, 0].imshow(inter_img_cubic)
    axs[0, 0].title("Bicubic")
    axs[0, 0].xlabel(f"PSNR = {psnr_cubic:.2f} dB")

    axs[0, 0].subplot(2, 5, 9)
    axs[0, 0].imshow(ssim_cubic, vmin=0, vmax=1)
    axs[0, 0].title("SSIM - Bicubic")
    axs[0, 0].xlabel(f"\nMSSIM = {mssim_cubic:.4f}")

    axs[0, 0].subplot(2, 5, 5)
    axs[0, 0].imshow(inter_img_lanczos4)
    axs[0, 0].title("Lanczos4")
    axs[0, 0].xlabel(f"PSNR = {psnr_lanczos4:.2f} dB")

    axs[0, 0].subplot(2, 5, 10)
    axs[0, 0].imshow(ssim_lanczos4, vmin=0, vmax=1)
    axs[0, 0].title("SSIM - Lanczos4")
    axs[0, 0].xlabel(f"\nMSSIM = {mssim_lanczos4:.4f}")

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.2, hspace=0.1)
    plt.show()
