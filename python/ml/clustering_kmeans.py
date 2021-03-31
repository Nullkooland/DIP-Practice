import numpy as np
import cv2
import pyheif
import matplotlib.pyplot as plt

# Number of clusters
K = 6

if __name__ == "__main__":
    # Read image and normalize to [0, 1] in fp32
    img_src = pyheif.read_as_numpy(
        "images/flower_field.heic").astype(np.float32) / 255.0

    # Convert to Lab color space and make vectorized copy
    img_lab = cv2.cvtColor(img_src, cv2.COLOR_RGB2LAB)
    img_vec = img_lab.reshape((-1, 3))

    # Apply K-Means on all pixels in Lab color
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, labels, centers = cv2.kmeans(
        img_vec, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Extract clusters data
    clusters = []
    for i in range(K):
        mask = (labels == i).flatten()
        cluster = (centers[i], img_vec[mask])
        clusters.append(cluster)

    clusters.sort(key=lambda cluster: len(cluster[1]))

    # Convert label map back to original image shape
    img_labels = centers[labels.flatten()].reshape((img_src.shape))

    # Convert back to RGB color space for dispaly
    img_labels = cv2.cvtColor(img_labels, cv2.COLOR_LAB2RGB)

    # Setup plots
    fig: plt.Figure = plt.figure(figsize=(16, 5))

    # Show source image
    ax0: plt.Axes = fig.add_subplot(1, 3, 1)
    ax0.imshow(img_src)
    ax0.set_title("Source")

    # Show labels
    ax1: plt.Axes = fig.add_subplot(1, 3, 2)
    ax1.imshow(img_labels)
    ax1.set_title("Labels")

    # Show clusters
    ax2: plt.Axes = fig.add_subplot(1, 3, 3, projection="3d")

    for i in range(K):
        center, data = clusters[i]
        color_center = cv2.cvtColor(center.reshape((1, 1, 3)), cv2.COLOR_LAB2RGB).squeeze()
        color_points = cv2.cvtColor(data.reshape(-1, 1, 3), cv2.COLOR_LAB2RGB).squeeze()
        
        ax2.scatter(data[:, 0], data[:, 1], data[:, 2],
                    marker=".", s=2, color=color_points, alpha=0.1)
        ax2.scatter(center[0], center[1], center[2],
                    marker="*", color=color_center, label=f"{i}")
        ax2.text(center[0] + 1e-2, center[1] + 1e-2, center[2] + 1e-2, f"{i}")

    ax2.legend()
    ax2.set_title("Color clusters")

    plt.show()
