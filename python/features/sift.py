from typing import List
import cv2
import numpy as np
from utils.image_reader import ImageReader
from visualizer import plot_keypoints, plot_matches

NUM_FEATURES = 1024

if __name__ == "__main__":
    # read two images of the same scene
    reader = ImageReader()
    img_src_0 = reader.read("images/sleepy_cat_scale_1.heic")
    img_src_1 = reader.read("images/sleepy_cat_scale_2.heic")
    mask_0 = reader.read("images/sleepy_cat_mask_scale_1.heic")
    mask_1 = reader.read("images/sleepy_cat_mask_scale_2.heic")

    mask_0 = cv2.cvtColor(mask_0, cv2.COLOR_RGB2GRAY)
    mask_1 = cv2.cvtColor(mask_1, cv2.COLOR_RGB2GRAY)

    # extract features using SIFT
    sift = cv2.SIFT_create(nfeatures=NUM_FEATURES)

    keypoints_0, desc_0 = sift.detectAndCompute(img_src_0, mask_0)
    keypoints_1, desc_1 = sift.detectAndCompute(img_src_1, mask_1)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=64)  # or pass empty dictionary
    matcher = cv2.FlannBasedMatcher(index_params, search_params)
    matches = matcher.knnMatch(desc_0, desc_1, k=2, compactResult=True)

    # do ratio test
    matches_valid: List[cv2.DMatch] = []

    for (match_primary, match_secondary) in matches:
        if match_primary.distance < 0.7 * match_secondary.distance:
            matches_valid.append(match_primary)

    plot_keypoints(img_src_0, keypoints_0)
    plot_matches(img_src_0, img_src_1, keypoints_0, keypoints_1, matches_valid)
