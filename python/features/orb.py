import cv2
import pyheif
import numpy as np
import matplotlib.pyplot as plt
import pyheif

if __name__ == "__main__":
    # Read source image pair
    img_src_0 = pyheif.read_as_numpy("./images/sleepy_cat_scale_0.heic")
    img_src_1 = pyheif.read_as_numpy("./images/sleepy_cat_scale_1.heic")
    mask_1 = pyheif.read_as_numpy(
        "./images/sleepy_cat_mask_scale_1.heic")
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
    plt.figure("ORB Matches", figsize=(14, 6))
    plt.imshow(img_matches)

    plt.show()
