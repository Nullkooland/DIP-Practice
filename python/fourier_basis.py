import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid import ImageGrid


def get_fourier_basis(size, u, v):
    # get 2D grid
    x = np.arange(size[0])
    y = np.arange(size[1])
    [xx, yy] = np.meshgrid(x, y)
    # calculate normalized frequency
    freq_x = np.float32(u) / size[0]
    freq_y = np.float32(v) / size[1]
    # generate fourier basis
    return np.exp(1j * 2 * np.pi * (freq_x * xx + freq_y * yy))


size = (512, 512)
uv = [
    (1, 0),
    (0, 1),
    (1, 1),
    (3, 1),
    (-3, 1),
    (4, 8)
]

plt.figure('2D Fourier Basis')

for i, (u, v) in enumerate(uv):
    plt.subplot(2, 3, i + 1)
    basis = get_fourier_basis(size, u, v)
    plt.imshow(np.real(basis), cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.title(f'u = {u}, v = {v}')

plt.show()