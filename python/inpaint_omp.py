import numpy as np
import scipy.sparse.linalg as sparse
import cv2
import matplotlib.pyplot as plt


def omp(A, b, max_iters=64, err_tol=1e-5):
    m, n = A.shape
    # Get adjoint operator of A
    A_adj = A.adjoint()

    # Init solution x to zero vector
    x = np.zeros(n)
    # Init residual r to b
    r = b.copy()
    # Init support set to empty set
    support_set = np.empty(0, dtype=np.int)

    norm_b = np.linalg.norm(b)
    err = np.finfo(np.float32).max
    i = 0

    while i < max_iters and err > err_tol:
        # Select the atom that most correlated to residual
        j = np.argmax(np.abs(A_adj @ r))
        # update support set
        support_set = np.append(support_set, j)

        # Create select operator S (x -> x_supp)
        # Its adjoint is expand operator S_adj (x_supp -> x)
        def expand_support_to_full(x_supp):
            x_full = np.zeros(n)
            x_full[support_set] = x_supp
            return x_full

        S = sparse.LinearOperator((len(support_set), n),
                                  matvec=lambda x: x[support_set],
                                  rmatvec=expand_support_to_full)

        # Project b to column space of A_supp
        # Solve the least-square problem min ||A(S_adj(x_supp)) - b||^2
        x_supp = sparse.lsmr(A @ S.adjoint(), b)[0]
        # Update x
        x[:] = 0
        x[support_set] = x_supp
        # Make residual orthogonal to column space of A_supp
        r = b - A @ x
        # Calculate relative error
        err = np.linalg.norm(r) / norm_b
        i += 1

    return x, support_set


def poke_holes(img, p):
    m, n = img.shape[:2]
    holes = np.random.rand(m, n) < p
    broken_img = img.copy()
    broken_img[holes] = 0
    return broken_img, holes


if __name__ == "__main__":
    # Load image
    src_img = cv2.imread('./images/cartoon.png')
    src_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
    src_img = np.float32(src_img) / 255.0
    m, n = src_img.shape

    # Generate image with missing pixels
    broken_img, holes = poke_holes(src_img, 0.3)

    # Create LinearOperator of missing pixels
    def missing_pixels(img_vec):
        img = np.reshape(img_vec, (m, n))
        img_broken = img.copy()
        img_broken[holes] = 0
        return img_broken.flatten()

    A = sparse.LinearOperator(dtype=np.float32, shape=(m * n, m * n),
                              matvec=missing_pixels, rmatvec=missing_pixels)

    # Take advantage of sparsity in transform domian
    # Create LinearOperator of forward-transform and inverse-transform
    def vec_dct(img_vec):
        img = np.reshape(img_vec, (m, n))
        d = cv2.dct(img)
        return d.flatten()

    def vec_idct(d_vec):
        d = np.reshape(d_vec, (m, n))
        img = cv2.idct(d)
        return img.flatten()

    C = sparse.LinearOperator(dtype=np.float32, shape=(m * n, m * n),
                              matvec=vec_dct, rmatvec=vec_idct)

    rec_d_vec, support_set = omp(
        A @ C.adjoint(), broken_img.flatten(), 5000, 1e-2)
    rec_img_vec = vec_idct(rec_d_vec)
    rec_img = np.reshape(rec_img_vec, (m, n))

    # Plot the broken vs. recovered results
    fig, axs = plt.subplots(1, 2, num='Result', figsize=(12, 6))

    axs[0].imshow(broken_img)
    axs[0].set_title('Broken')

    axs[1].imshow(rec_img)
    axs[1].set_title('Recovered')

    plt.show()
