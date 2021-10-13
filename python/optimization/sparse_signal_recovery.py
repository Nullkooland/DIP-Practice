import numpy as np
from numpy.typing import NDArray
from typing import Tuple
from scipy import sparse
from scipy.linalg import solve
from scipy.sparse.linalg.interface import LinearOperator
from scipy.sparse.linalg.isolve import cg
import matplotlib.pyplot as plt

N = 128     # Dimension of original signal
M = 32      # DImension of sampled signal
S = 10      # Sparsity of original signal


def soft_threshold(x: np.ndarray, threshold: float) -> np.ndarray:
    return np.maximum(np.abs(x) - threshold, 0.0) * np.sign(x)


def basis_pursuit_admm(A: np.ndarray, y: np.ndarray, x0: np.ndarray,
                       rho: float = 100, tol_err: float = 1e-3, max_iters: int = 64):
    m, n = A.shape
    x = x0.copy()
    z = x0.copy()
    u = np.zeros_like(x0)
    err = np.full((max_iters,), np.nan, dtype=np.float32)

    AAt = A @ A.T
    I = np.eye(n, dtype=np.float32)
    P = I - A.T @ solve(AAt, A, assume_a="pos")
    q = A.T @ solve(AAt, y, assume_a="pos")

    i = 0
    while i < max_iters:
        x = P @ (z - u) + q
        z = soft_threshold(x + u, 1 / rho)
        u += x - z

        err[i] = np.sum(np.abs(x))
        if err[i] < tol_err:
            break

        i += 1

    return x, i, err


def lasso_admm(A: np.ndarray, y: np.ndarray, x0: np.ndarray, lam: float, rho: float = 100, tol_err: float = 1e-3, max_iters: int = 64):
    m, n = A.shape
    x = x0.copy()
    z = x0.copy()
    w = np.zeros_like(x0)

    I = np.eye(n, dtype=np.float32)
    err = np.full((max_iters,), np.nan, dtype=np.float32)

    P = A.T @ A + rho * I
    q = A.T @ y

    i = 0
    while i < max_iters:
        x = solve(P, q + rho * (z - w), assume_a="pos")
        z = soft_threshold(x + w, lam / rho)
        w += x - z

        err[i] = 0.5 * np.sum((A @ x - y) ** 2) + lam * np.sum(np.abs(x))
        if err[i] < tol_err:
            break

        i += 1

    return x, i, err


def slove_lasso_admm(
        A: LinearOperator,
        y: NDArray[np.float32],
        x_init: NDArray[np.float32],
        lam: float,
        rho: float = 100,
        tol_err: float = 1e-3,
        max_iters: int = 64) -> Tuple[NDArray[np.float32], int, NDArray[np.float32]]:
    # Shrinkage operator
    # Prepare variables used in iteration
    x = x_init.copy()
    z = x_init.copy()
    w = np.zeros_like(x_init)

    n = x.shape[0]

    err = np.full((max_iters,), np.nan, dtype=np.float32)
    rhoI = rho * np.eye(n, dtype=np.float32)

    # Precompute some middle variables
    # rhoI = LinearOperator(
    #     dtype=np.float32,
    #     shape=A.shape,
    #     matvec=lambda img_vec: rho * img_vec,
    #     rmatvec=lambda img_vec: rho * img_vec
    # )

    P = A.T @ A + rhoI
    q = A.T @ y

    i = 0
    while i < max_iters:
        x, ret = cg(P, q + rho * (z - w))
        z = soft_threshold(x + w, lam / rho)
        w += x - z

        err[i] = 0.5 * np.sum((A @ x - y) ** 2) + lam * np.sum(np.abs(x))
        if err[i] < tol_err:
            break

        i += 1

    return x, i, err


if __name__ == "__main__":
    # Generate indices of non-zero entries
    supp = np.random.randint(0, N, size=(S, ))
    # Generate original sparse signal
    x = np.zeros((N, ), dtype=np.float32)
    x[supp] = np.random.rand(S) * 0.5 + 0.25

    # Generate sampling matrix
    # A = np.random.randn(M, N) / np.sqrt(M)

    mask = np.random.rand(M, N) > 0.5
    A = np.full((M, N), 1 / np.sqrt(M))
    A[mask] *= -1

    # Get sampled signal
    y = A @ x

    # Recover the original signal
    # Initial guess
    x0 = A.T @ y

    # x_rec, num_iters, err = slove_lasso_admm(A, y, x0, 0.25, 20, 1e-2, 400)
    x_rec, num_iters, err = basis_pursuit_admm(A, y, x0, 20, 1e-2, 400)

    fig, axs = plt.subplots(1, 2, num="ADMM recovery", figsize=(12, 6))
    axs[0].stem(x, basefmt="None", linefmt="C0-",
                markerfmt="C0o", label=r"$x[n]$")
    axs[0].stem(x_rec, basefmt="None", linefmt="C3-",
                markerfmt="C3x", label=r"$\hat{x}[n]$")

    axs[1].plot(err[:num_iters])
    axs[1].set_yscale("log")

    axs[0].legend(loc="upper right")
    plt.show()
