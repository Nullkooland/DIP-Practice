import cv2
import numpy as np
from typing import List
from utils.image_reader import ImageReader
from visualizer import plot_keypoints, plot_matches

if __name__ == "__main__":
    # Read source image pair
    reader = ImageReader()
    img_src_0 = reader.read("images/eaidk310_0.heic")
    img_src_1 = reader.read("images/eaidk310_0.heic")

    # Get keypoints and feature descriptors using AKAZE algorithm
    akaze = cv2.AKAZE_create(threshold=9e-3)
    keypoints_0, descriptors_0 = akaze.detectAndCompute(img_src_0, None)
    keypoints_1, descriptors_1 = akaze.detectAndCompute(img_src_1, None)

    # Match descriptors of two images
    matcher = cv2.BFMatcher_create(normType=cv2.NORM_HAMMING)
    matches = matcher.knnMatch(
        descriptors_0, descriptors_1, k=2, compactResult=True)

    # do ratio test
    matches_valid: List[cv2.DMatch] = []

    for (match_primary, match_secondary) in matches:
        if match_primary.distance < 0.7 * match_secondary.distance:
            matches_valid.append(match_primary)

    plot_keypoints(img_src_0, keypoints_0)
    plot_matches(img_src_0, img_src_1, keypoints_0, keypoints_1, matches_valid)
