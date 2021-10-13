import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import cv2
from utils.image_reader import ImageReader


def generateMandelbrotSet(width, height, gen_range, cmap=cm.hot(range(256))):
    # get generate range
    x0, x1, y0, y1 = gen_range

    # get x, y range
    x = np.linspace(x0, x1, width)
    y = np.linspace(y0, y1, height)

    # get 2d grid of complex numbers
    real, imag = np.meshgrid(x, y, sparse=True)
    c = real + imag * 1.0j

    # prepare iteration
    z = np.copy(c)
    notdone = np.ones(z.shape, dtype=np.bool)
    iter_times = np.zeros(z.shape, dtype=np.int32)
    max_iter_times = cmap.shape[0]

    # vectorized iteration
    for _ in range(max_iter_times):
        notdone = np.abs(z) < 4.0
        iter_times[notdone] += 1

        z[notdone] = z[notdone]**2 + c[notdone]

    # for converge values set as cmap[0]
    iter_times[notdone] = 0

    # generate image
    return np.take(cmap, iter_times, axis=0)


if __name__ == "__main__":
    width = 1024
    height = 1024
    colormap = cm.hot(range(512))
    # gen_range = (-2.0, 0.5, -1.25, 1.25)
    gen_range = (0.34, 0.36, 0.40, 0.42)

    img = generateMandelbrotSet(width, height, gen_range, colormap)
    plt.imshow(img, extent=gen_range, interpolation="lanczos")
    plt.show()

    # img = np.uint8(img * 255.0)
    # img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    # cv2.imwrite("./output/mandelbrot_set.png", img)
