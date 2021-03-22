import cv2
import pyheif
import numpy as np
import matplotlib.pyplot as plt
import pyheif

if __name__ == "__main__":
    # Read source image pair
    img_src_0 = pyheif.read_as_numpy("./images/eaidk310_0.heic")
    img_src_1 = pyheif.read_as_numpy("./images/eaidk310_1.heic")

    # Get keypoints and feature descriptors using BRISK algorithm
    brisk = cv2.BRISK_create(thresh=120)
    keypoints_0, descriptors_0 = brisk.detectAndCompute(img_src_0, None)
    keypoints_1, descriptors_1 = brisk.detectAndCompute(img_src_1, None)

    # Match descriptors of two images
    matcher = cv2.BFMatcher_create(normType=cv2.NORM_HAMMING)
    matches = matcher.knnMatch(
        descriptors_0, descriptors_1, k=2, compactResult=True)

    # Do ratio test
    mask_matches = np.zeros((len(matches), 2), dtype=np.uint8)
    for (match_primary, match_secondary) in matches:
        if match_primary.distance < 0.7 * match_secondary.distance:
            i = match_primary.queryIdx
            mask_matches[i, 0] = 1

    # Draw matches
    img_matches = cv2.drawMatchesKnn(
        img_src_0, keypoints_0,
        img_src_1, keypoints_1,
        matches, None,
        matchesMask=mask_matches,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Show matches
    plt.figure("BRISK Matches", figsize=(14, 6))
    plt.imshow(img_matches)

    plt.show()
