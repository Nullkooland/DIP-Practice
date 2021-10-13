from typing import List

import cv2
import numpy as np
from utils.image_reader import ImageReader
from visualizer import plot_keypoints, plot_matches

if __name__ == "__main__":
    # Read source image pair
    reader = ImageReader()
    img_src_0 = reader.read("images/sleepy_cat_scale_0.heic")
    img_src_1 = reader.read("images/sleepy_cat_scale_1.heic")
    mask_1 = reader.read("images/sleepy_cat_mask_scale_1.heic")
    mask_1 = cv2.cvtColor(mask_1, cv2.COLOR_RGB2GRAY)

    # Get keypoints and feature descriptors using ORB algorithm
    orb = cv2.ORB_create(nfeatures=1024, fastThreshold=30)
    keypoints_0, descriptors_0 = orb.detectAndCompute(img_src_0, None)
    keypoints_1, descriptors_1 = orb.detectAndCompute(img_src_1, mask_1)

    # Match descriptors of two images
    FLANN_INDEX_LSH = 6
    index_params = dict(algorithm=FLANN_INDEX_LSH,
                        table_number=16, key_size=16)
    search_params = dict(checks=64)  # or pass empty dictionary
    matcher = cv2.FlannBasedMatcher(index_params, search_params)
    matches = matcher.knnMatch(
        descriptors_0, descriptors_1, k=2, compactResult=True)

    # do ratio test
    matches_valid: List[cv2.DMatch] = []

    for (match_primary, match_secondary) in matches:
        if match_primary.distance < 0.7 * match_secondary.distance:
            matches_valid.append(match_primary)

    # plot_keypoints(img_src_1, keypoints_1)
    plot_matches(img_src_0, img_src_1, keypoints_0, keypoints_1, matches_valid)
