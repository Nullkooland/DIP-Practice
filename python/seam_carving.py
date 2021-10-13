import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
from utils.image_reader import ImageReader


def get_energy(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Calculate image energy
    gx = cv2.Sobel(gray_img, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(gray_img, cv2.CV_32F, 0, 1)
    energy = np.abs(gx) + np.abs(gy)
    return energy


def get_cme(energy):
    (h, w) = energy.shape
    # Calculate vertical cumulative minimum energy
    M = np.zeros((h, w + 2))
    # Init first row
    M[0, 1:-1] = energy[0, :]
    # Pad two columns for edge case
    M[:, 0] = float("Inf")
    M[:, -1] = float("Inf")

    max_cme = 0
    for i in range(1, h):
        for j in range(w):
            cme_l = M[i - 1, j]  # left diagonal
            cme_u = M[i - 1, j + 1]  # up
            cme_r = M[i - 1, j + 2]  # right diagonal
            M[i, j + 1] = energy[i, j] + min(min(cme_l, cme_r), cme_u)
            max_cme = max(M[i, j + 1], max_cme)

    return (M, max_cme)


def get_seam(M):
    # Backtrack from bottom to top
    (h, _) = M.shape
    s = np.zeros(h, np.int32)
    s[h - 1] = np.argmin(M[-1, :])  # start x position

    for i in reversed(range(1, h)):
        dx = np.argmin(M[i - 1, (s[i] - 1):(s[i] + 2)]) - 1
        s[i - 1] = s[i] + dx

    return s - 1  # Remove the offset


def delete_seam(I, s):
    (h, w, _) = I.shape
    mask = np.ones(I.shape, np.bool)

    for i in range(h):
        mask[i, s[i]] = False

    return I[mask].reshape(h, w - 1, 3)


def add_seam(I, s):
    (h, _, _) = I.shape

    for i in range(h):
        I[i, (s[i] + 1):] = I[i, s[i]:-1]
        I[i, s[i]] = (I[i, s[i]] + I[i, s[i] - 1]) / 2

    return I


def shift_seams(ns, k):
    n = ns.shape[0]
    current = ns[k, :]

    for l in range(k + 1, n):
        remaining = ns[l, :]
        remaining[remaining >= current] += 2

    return ns


def enlarge(src_img, dw):
    (h, w, _) = src_img.shape
    ns = np.zeros((dw, h), np.int32)
    # First delete n times to get n seams
    I = np.copy(src_img)
    for k in range(dw):
        E = get_energy(I)
        (M, _) = get_cme(E)
        ns[k, :] = get_seam(M)
        I = delete_seam(I, ns[k, :])

    # Insert the seams found into original image
    I = cv2.copyMakeBorder(src_img, 0, 0, 0, dw, cv2.BORDER_CONSTANT)
    for k in range(dw):
        I = add_seam(I, ns[k, :])
        ns = shift_seams(ns, k)

    return I


if __name__ == "__main__":
    reader = ImageReader()
    src_img = reader.read("images/autumn.heic", np.float32)

    I = np.copy(src_img)
    # I = np.rot90(src_img, 1)

    b = time.time()
    for _ in range(64):
        E = get_energy(I)
        (M, _) = get_cme(E)
        s = get_seam(M)
        I = delete_seam(I, s)
    e = time.time()
    print(e - b)

    # I = np.rot90(I, -1)

    # I = enlarge(src_img, 384)

    plt.figure("Seam Carving")

    plt.subplot(1, 2, 1)
    plt.title("Original")
    plt.imshow(src_img)

    plt.subplot(1, 2, 2)
    plt.title("Result")
    plt.imshow(I)

    plt.show()
