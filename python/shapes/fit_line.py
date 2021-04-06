import cv2
import pyheif
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

SLOPE = 1.25
Y_OFFSET = -0.5
NUM_POINTS = 128
NOISE_STD = 0.25

NUM_OUTLIERS = NUM_POINTS // 16
OUTLIER_Y_MEAN_OFFSET = 2.5

FIT_DIST_TYPES = [
    ("L1", cv2.DIST_L1),
    ("L2", cv2.DIST_L2),
    ("L12", cv2.DIST_L12),
    ("FAIR", cv2.DIST_FAIR),
    ("WELSCH", cv2.DIST_WELSCH),
    ("HUBER", cv2.DIST_HUBER),
]

if __name__ == "__main__":
    x = np.random.rand(NUM_POINTS)
    x = np.sort(x)
    n = np.random.randn(NUM_POINTS) * NOISE_STD
    y = SLOPE * x + Y_OFFSET + n

    outlier_indices = np.random.randint(0, NUM_POINTS // 4, NUM_OUTLIERS)
    y[outlier_indices] += np.random.randn(NUM_OUTLIERS) * NOISE_STD + OUTLIER_Y_MEAN_OFFSET
    observations = np.stack((x, y), axis=1)

    plt.figure("Line fitting", figsize=(8, 8))
    # show observations
    plt.plot(x, y, "o", color="black", label="data")
    # highlight outlier
    plt.plot(x[outlier_indices], y[outlier_indices],
             "o", color="red", label="outlier")
    # draw groundtruth line
    plt.axline((0, Y_OFFSET), slope=SLOPE,
               linestyle="--", linewidth=2, color="gray", label="groundtruth")

    cmap = get_cmap("tab10")

    for i, (dist_type_name, dist_type) in enumerate(FIT_DIST_TYPES):
        line = cv2.fitLine(observations, dist_type, 0, 0.01, 0.01)
        vx, vy, x0, y0 = line.squeeze()
        plt.axline((x0, y0), slope=vy / vx,
                   label=f"Fit-{dist_type_name}", color=cmap.colors[i])

    plt.ylim(np.min(y) * 1.5, np.max(y) * 1.5)
    plt.legend(loc="upper right")
    plt.show()
