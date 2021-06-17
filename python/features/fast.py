import cv2
import pyheif
from visualizer import plot_keypoints

if __name__ == "__main__":
    # Read image
    img_src = pyheif.read_as_numpy("./images/boat.heic")
    img_gray = cv2.cvtColor(img_src, cv2.COLOR_RGB2BGRA)

    # Create FAST feature detector
    fast = cv2.FastFeatureDetector_create(
        threshold=64, nonmaxSuppression=True, type=cv2.FAST_FEATURE_DETECTOR_TYPE_9_16)

    # Detect and visualize blobs
    keypoints = fast.detect(img_gray)
    plot_keypoints(img_src, keypoints)
