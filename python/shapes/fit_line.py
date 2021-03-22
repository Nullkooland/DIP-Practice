import cv2
import pyheif
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

SLOPE = 1.25
Y_OFFSET = -0.5
NUM_POINTS = 100
NOISE_STD = 0.25
OUTLIER_Y_OFFSET = 5.0

FIT_DIST_TYPES = [
    ("L1", cv2.DIST_L1),
    ("L2", cv2.DIST_L2),
    ("L12", cv2.DIST_L12),
    ("FAIR", cv2.DIST_FAIR),
    ("WELSCH", cv2.DIST_WELSCH),
    ("HUBER", cv2.DIST_HUBER),
]

if __name__ == "__main__":
    x= np.random.rand(NUM_POINTS)
    n= np.random.randn(NUM_POINTS) * NOISE_STD
    y= SLOPE * x + Y_OFFSET + n

    outlier_index= np.random.randint(0, NUM_POINTS)
    y[outlier_index] += OUTLIER_Y_OFFSET
    observations= np.stack((x, y), axis=1)

    plt.figure("Line fitting", figsize=(8, 8))
    # show observations
    plt.plot(x, y, "o", color="black")
    # highlight outlier
    plt.plot(x[outlier_index], y[outlier_index], "o", color="red")
    # draw groundtruth line
    plt.axline((0, Y_OFFSET), slope=SLOPE,
               linestyle="--", linewidth=2, color="gray", label="groundtruth")
               
    cmap = get_cmap("tab10")

    for i, (dist_type_name, dist_type) in enumerate(FIT_DIST_TYPES):
        line=cv2.fitLine(observations, dist_type, 0, 0.01, 0.01)
        vx, vy, x0, y0=line.squeeze()
        plt.axline((x0, y0), slope=vy / vx, label=dist_type_name, color=cmap.colors[i])

    plt.ylim(np.min(y) * 1.5, np.max(y) * 1.5)
    plt.legend()
    plt.show()
