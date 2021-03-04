import cv2
import numpy as np
import matplotlib.pyplot as plt
import pyheif

if __name__ == "__main__":
    # Read image
    img_src = pyheif.read_as_numpy("./images/boat.heic")
    img_gray = cv2.cvtColor(img_src, cv2.COLOR_RGB2BGRA)

    # Create FAST feature detector
    fast = cv2.FastFeatureDetector_create(
        threshold=64, nonmaxSuppression=True, type=cv2.FAST_FEATURE_DETECTOR_TYPE_9_16)

    # Detect and visualize blobs
    keypoints = fast.detect(img_gray)

    img_anno = cv2.drawKeypoints(
        img_src, keypoints, None, (255, 0, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    fig, (ax_src, ax_anno) = plt.subplots(
        1, 2, num="Blobs", sharey=True, figsize=(14, 6))

    ax_src.imshow(img_src)
    ax_anno.imshow(img_anno)

    ax_src.set_title("Original")
    ax_anno.set_title("FAST keypoints")

    plt.show()
