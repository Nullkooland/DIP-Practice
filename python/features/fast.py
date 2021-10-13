import cv2
from visualizer import plot_keypoints
from utils.image_reader import ImageReader

if __name__ == "__main__":
    # Read image
    reader = ImageReader()
    img_src = reader.read("images/boat.heic")
    img_gray = cv2.cvtColor(img_src, cv2.COLOR_RGB2GRAY)

    # Create FAST feature detector
    fast = cv2.FastFeatureDetector_create(
        threshold=64, nonmaxSuppression=True, type=cv2.FAST_FEATURE_DETECTOR_TYPE_9_16)

    # Detect and visualize blobs
    keypoints = fast.detect(img_gray)
    plot_keypoints(img_src, keypoints)
