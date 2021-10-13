import numpy as np
import cv2
from numpy.core.fromnumeric import shape
import pyheif
import matplotlib.pyplot as plt
import scipy.sparse.linalg as sparse_linalg
from scipy.sparse.linalg import LinearOperator


def poke_holes(img, p):
    m, n = img.shape[:2]
    holes = np.random.rand(m, n) < p
    broken_img = img.copy()
    broken_img[holes] = 0
    return broken_img, holes


if __name__ == "__main__":
    src_img = pyheif.read_as_numpy("./images/lena.heic")
    src_img = np.float32(src_img) / 255.0
    l_tv = 1e-3

    m, n, c = src_img.shape

    broken_img, holes = poke_holes(src_img, 0.5)

    def missing_pixels(img_vec):
        img = np.reshape(img_vec, (m, n, c))
        img_broken = img.copy()
        img_broken[holes] = 0
        return img_broken.flatten()

    # TV regularizer
    def diff_x(img_vec):
        img = np.reshape(img_vec, (m, n, c))
        diff_x = np.diff(img, axis=1)
        return diff_x.flatten()

    def diff_x_adjoint(diff_x_vec):
        diff_x = np.reshape(diff_x_vec, (m, n - 1, c))
        zero_col = np.zeros((m, 1, c), dtype=diff_x.dtype)
        img = np.hstack((zero_col, diff_x)) - np.hstack((diff_x, zero_col))
        return img.flatten()

    def diff_y(img_vec):
        img = np.reshape(img_vec, (m, n, c))
        diff_y = np.diff(img, axis=0)
        return diff_y.flatten()

    def diff_y_adjoint(diff_y_vec):
        diff_y = np.reshape(diff_y_vec, (m - 1, n, c))
        zero_row = np.zeros((1, m, c), dtype=diff_y.dtype)
        img = np.vstack((zero_row, diff_y)) - np.vstack((diff_y, zero_row))
        return img.flatten()

    A = LinearOperator(
        dtype=np.float32,
        shape=(m*n*c, m*n*c),
        matvec=lambda img_vec:
        missing_pixels(missing_pixels(img_vec)) +
        l_tv * (
            diff_x_adjoint(diff_x(img_vec)) +
            diff_y_adjoint(diff_y(img_vec))
        )
    )

    b = broken_img.flatten()

    rec_img_vec, info = sparse_linalg.cgs(A, b, maxiter=64)
    rec_img = np.reshape(rec_img_vec, (m, n, c))

    # Plot the broken vs. recovered results
    fig, axs = plt.subplots(1, 2, num="Result", figsize=(12, 6))

    axs[0].imshow(broken_img)
    axs[0].set_title("Broken")

    axs[1].imshow(rec_img)
    axs[1].set_title("Recovered")

    plt.show()