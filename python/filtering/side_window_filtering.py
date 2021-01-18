import cv2
import numpy as np


def add_salt_and_pepper(img, p):
    h, w, c = img.shape
    noised = img.copy()

    num_noise = int(h * w * p / 2)
    coords_salt = [np.random.randint(0, length, num_noise)
                   for length in (h, w)]
    coords_pepper = [np.random.randint(
        0, length, num_noise) for length in (h, w)]

    noised[coords_salt[0], coords_salt[1], :] = 1
    noised[coords_pepper[0], coords_pepper[1], :] = 0
    return noised


def add_gaussian_noise(img, std):
    h, w, c = img.shape
    noised = img.copy()
    noise = np.random.randn(h, w, c) * std
    noised += noise
    np.clip(noised, 0, 1, out=noised)
    return noised


def get_side_windows(kernel):
    k = kernel.shape[0]
    k_half = (k + 1) // 2

    kernel_L = kernel.copy()
    kernel_L[k_half:] = 0.0
    kernel_L /= np.sum(kernel_L)
    kernel_R = np.flip(kernel_L)

    windows = []
    windows.append((kernel_L, kernel))
    windows.append((kernel_R, kernel))
    windows.append((kernel, kernel_L))
    windows.append((kernel, kernel_R))
    windows.append((kernel_L, kernel_L))
    windows.append((kernel_L, kernel_R))
    windows.append((kernel_R, kernel_L))
    windows.append((kernel_R, kernel_R))

    return windows


def swf_linear(img, kernel):
    h, w, c = img.shape
    filtered = np.empty((8, h, w, c), dtype=np.float32)
    error = np.empty((8, h, w), dtype=np.float32)

    for i, (kernel_x, kernel_y) in enumerate(get_side_windows(kernel)):
        filtered[i] = cv2.sepFilter2D(
            img, cv2.CV_32F, kernel_x, kernel_y)
        error[i] = np.linalg.norm(filtered[i] - img, axis=2)

    select_side_index = np.argmin(error, axis=0)
    result = np.empty((h, w, c), dtype=np.float32)

    result[..., 0] = np.choose(select_side_index, filtered[..., 0])
    result[..., 1] = np.choose(select_side_index, filtered[..., 1])
    result[..., 2] = np.choose(select_side_index, filtered[..., 2])

    return result


def swf_median(img, k):
    """Side Windowed Median Blur

    This is slow as shit
    """
    h, w, c = img.shape
    p = (k - 1) // 2
    padded = cv2.copyMakeBorder(img, p, p, p, p, cv2.BORDER_REFLECT101)

    median = np.empty((8, h, w, c), dtype=np.uint8)
    error = np.empty((8, h, w), dtype=np.float32)

    for i in range(h):
        for j in range(w):
            roi_L = padded[i:i + 2*p + 1, j:j + p + 1]
            median[0, i, j, 0] = np.median(roi_L[:, :, 0])
            median[0, i, j, 1] = np.median(roi_L[:, :, 1])
            median[0, i, j, 2] = np.median(roi_L[:, :, 2])

            roi_R = padded[i:i + 2*p + 1, j + p:j + 2*p + 1]
            median[1, i, j, 0] = np.median(roi_R[:, :, 0])
            median[1, i, j, 1] = np.median(roi_R[:, :, 1])
            median[1, i, j, 2] = np.median(roi_R[:, :, 2])

            roi_U = padded[i:i + p + 1, j:j + 2*p + 1]
            median[2, i, j, 0] = np.median(roi_U[:, :, 0])
            median[2, i, j, 1] = np.median(roi_U[:, :, 1])
            median[2, i, j, 2] = np.median(roi_U[:, :, 2])

            roi_D = padded[i+p:i + 2*p + 1, j:j + 2*p + 1]
            median[3, i, j, 0] = np.median(roi_D[:, :, 0])
            median[3, i, j, 1] = np.median(roi_D[:, :, 1])
            median[3, i, j, 2] = np.median(roi_D[:, :, 2])

            roi_LU = padded[i:i+p+1, j:j + p + 1]
            median[4, i, j, 0] = np.median(roi_LU[:, :, 0])
            median[4, i, j, 1] = np.median(roi_LU[:, :, 1])
            median[4, i, j, 2] = np.median(roi_LU[:, :, 2])

            roi_LD = padded[i+p:i+2*p+1, j:j + p + 1]
            median[5, i, j, 0] = np.median(roi_LD[:, :, 0])
            median[5, i, j, 1] = np.median(roi_LD[:, :, 1])
            median[5, i, j, 2] = np.median(roi_LD[:, :, 2])

            roi_RU = padded[i:i+p+1, j+p:j + 2*p + 1]
            median[6, i, j, 0] = np.median(roi_RU[:, :, 0])
            median[6, i, j, 1] = np.median(roi_RU[:, :, 1])
            median[6, i, j, 2] = np.median(roi_RU[:, :, 2])

            roi_RD = padded[i+p:i+2*p+1, j+p:j + 2*p + 1]
            median[7, i, j, 0] = np.median(roi_RD[:, :, 0])
            median[7, i, j, 1] = np.median(roi_RD[:, :, 1])
            median[7, i, j, 2] = np.median(roi_RD[:, :, 2])

    img = np.float32(img)
    for i in range(8):
        error[i] = np.linalg.norm(median[i] - img, axis=2)

    select_side_index = np.argmin(error, axis=0)
    result = np.empty((h, w, c), dtype=np.uint8)

    result[..., 0] = np.choose(select_side_index, median[..., 0])
    result[..., 1] = np.choose(select_side_index, median[..., 1])
    result[..., 2] = np.choose(select_side_index, median[..., 2])

    return result


src_img = cv2.imread('./images/opossum.png')
src_img = np.float32(src_img / 255.0)

gaussian_kernel = cv2.getGaussianKernel(7, 1.5, ktype=cv2.CV_32F)

box_kernel = np.ones((7, 1), np.float32)
box_kernel /= np.sum(box_kernel)

noised = add_salt_and_pepper(src_img, 0.075)
noised = add_gaussian_noise(noised, 0.05)

filtered_box = noised.copy()
filtered_box_swf = noised.copy()
filtered_gaussian = noised.copy()
filtered_gaussian_swf = noised.copy()
filtered_median = np.uint8(noised * 255.0)
# filtered_median_swf = np.uint8(noised * 255.0)
filtered_bilateral = noised.copy()

for _ in range(1):
    filtered_gaussian = cv2.sepFilter2D(
        filtered_gaussian, cv2.CV_32F, gaussian_kernel, gaussian_kernel)
    filtered_gaussian_swf = swf_linear(filtered_gaussian_swf, gaussian_kernel)
    filtered_box = cv2.boxFilter(filtered_box, cv2.CV_32F, (7, 7))
    filtered_box_swf = swf_linear(filtered_box_swf, box_kernel)
    filtered_median = cv2.medianBlur(filtered_median, 3)
    filtered_bilateral = cv2.bilateralFilter(filtered_bilateral, 7, 1.5, 0.5)

# filtered_median_swf = swf_median(filtered_median_swf, 7)

cv2.imshow('Original', src_img)
cv2.imshow('Noised', noised)

# cv2.imshow('Box', filtered_box)
# cv2.imshow('Gaussian', filtered_gaussian)
# cv2.imshow('Median', filtered_median)
# cv2.imshow('Bilateral', filtered_bilateral)
cv2.imshow('SWF - Box', filtered_box_swf)
cv2.imshow('SWF - Gaussian', filtered_gaussian_swf)
# cv2.imshow('SWF - Median', filtered_median_swf)
cv2.waitKey()

# cv2.imwrite('./output/noised_1.png', np.uint8(noised * 255.0))
# cv2.imwrite('./output/filtered_box_1.png', np.uint8(filtered_box * 255.0))
# cv2.imwrite('./output/filtered_gaussian_1.png', np.uint8(filtered_gaussian * 255.0))
# cv2.imwrite('./output/filtered_median_1.png', filtered_median)
# cv2.imwrite('./output/filtered_bilateral_1.png', np.uint8(filtered_bilateral * 255.0))
# cv2.imwrite('./output/filtered_box_swf_1.png', np.uint8(filtered_box_swf * 255.0))
# cv2.imwrite('./output/filtered_gaussian_swf_1.png', np.uint8(filtered_gaussian_swf * 255.0))
# cv2.imwrite('./output/filtered_median_swf.png', filtered_median_swf)
