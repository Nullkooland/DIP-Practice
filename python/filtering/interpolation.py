import numpy as np
import matplotlib.pyplot as plt

N = 64
D = 3
FREQ = 1

n_d = np.arange((N + D - 1) // D)
n = np.arange(N)

# x_d = np.cos(FREQ * n_d * 2 * np.pi / (N / D))
x_d = np.zeros((N + D - 1) // D)
x_d[:N // D // 2] = 1

# upsampling (zero-padded)
x = np.zeros(N)
x[n_d * D] = x_d

# interpolation kernels
linear_kernel_L = np.linspace(1 / D, 1, D)
linear_kernel = np.concatenate((linear_kernel_L, np.flip(linear_kernel_L[:-1])))

a = 3
lanczos_range_R = np.linspace(0, a, a * D)
lanczos_kernel_R = a * np.sin(np.pi * lanczos_range_R) * np.sin(
    np.pi*lanczos_range_R / a) / (np.pi*lanczos_range_R) ** 2

lanczos_kernel_R[0] = 1
lanczos_kernel = np.concatenate(
    (np.flip(lanczos_kernel_R[1:]), lanczos_kernel_R))

# interpolation
x_linear = np.convolve(x, linear_kernel, mode="same")
x_lanczos = np.convolve(x, lanczos_kernel, mode="same")

# x_linear = np.convolve(x, linear_kernel, mode="full")[D:D+N]
# x_lanczos = np.convolve(x, lanczos_kernel, mode="full")[a*D:a*D+N]

plt.figure("Interpolation", figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.stem(n, x, use_line_collection=True)
plt.xticks(n_d * D)
plt.yticks(np.arange(-1.2, 1.2, 0.4))
plt.grid(True, linestyle=":")
plt.title("Upsampling")

plt.subplot(3, 1, 2)
plt.stem(n, x_linear, use_line_collection=True)
plt.xticks(n_d * D)
plt.yticks(np.arange(-1.2, 1.2, 0.4))
plt.grid(True, linestyle=":")
plt.title("Linear Interpolation")

plt.subplot(3, 1, 3)
plt.stem(n, x_lanczos, use_line_collection=True)
plt.xticks(n_d * D)
plt.yticks(np.arange(-1.2, 1.2, 0.4))
plt.grid(True, linestyle=":")
plt.title("Lanczos Interpolation")

plt.tight_layout()
plt.show()
