import cv2
from visualizer import plot_keypoints
from utils.image_reader import ImageReader

if __name__ == "__main__":
    # Read image
    reader = ImageReader()
    img_src = reader.read("images/cloth.heic")
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

    plot_keypoints(img_src, blobs, show_response=False)