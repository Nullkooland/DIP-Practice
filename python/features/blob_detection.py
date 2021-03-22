import cv2
import pyheif
import numpy as np
import matplotlib.pyplot as plt
import pyheif

if __name__ == "__main__":
    # Read image
    img_src = pyheif.read_as_numpy("./images/cloth.heic")
    img_gray = cv2.cvtColor(img_src, cv2.COLOR_RGB2GRAY)

    # Set blob detector params
    params = cv2.SimpleBlobDetector_Params()

    # Set threshold range and step
    params.minThreshold = 50
    params.maxThreshold = 200
    params.thresholdStep = 10

    # Set overlapping factors
    params.minDistBetweenBlobs = 16
    params.minRepeatability = 2

    # Filter by color (keep only dark blobs)
    params.filterByColor = True
    params.blobColor = 0

    # Filter by area
    params.filterByArea = True
    params.minArea = 200

    # Filter by circularity
    params.filterByCircularity = True
    params.minCircularity = 0.3

    # Filter by convexity
    params.filterByConvexity = False
    params.minConvexity = 0.8

    # Filter by inertia ratio
    params.filterByInertia = True
    params.minInertiaRatio = 0.1

    # Create blob detector
    detector = cv2.SimpleBlobDetector_create(params)

    # Detect and visualize blobs
    blobs = detector.detect(img_src)

    img_anno = cv2.drawKeypoints(
        img_src, blobs, None, (255, 0, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    fig, (ax_src, ax_anno) = plt.subplots(
        1, 2, num="Blobs", sharey=True, figsize=(12, 6))

    ax_src.imshow(img_src)
    ax_anno.imshow(img_anno)

    ax_src.set_title("Original")
    ax_anno.set_title("Blobs")

    plt.show()
