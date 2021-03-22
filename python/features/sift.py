import cv2
import pyheif
import numpy as np
import matplotlib.pyplot as plt
import pyheif

NUM_FEATURES = 1024

if __name__ == "__main__":
    # read two images of the same scene
    img_src_1 = pyheif.read_as_numpy("./images/sleepy_cat_scale_1.heic")
    img_src_2 = pyheif.read_as_numpy("./images/sleepy_cat_scale_2.heic")
    mask_1 = pyheif.read_as_numpy(
        "./images/sleepy_cat_mask_scale_1.heic")
    mask_2 = pyheif.read_as_numpy(
        "./images/sleepy_cat_mask_scale_2.heic")

    mask_1 = cv2.cvtColor(mask_1, cv2.COLOR_RGB2GRAY)
    mask_2 = cv2.cvtColor(mask_2, cv2.COLOR_RGB2GRAY)

    # extract features using SIFT
    sift = cv2.SIFT_create(nfeatures=NUM_FEATURES)

    keypoints_1, desc_1 = sift.detectAndCompute(img_src_1, mask_1)
    keypoints_2, desc_2 = sift.detectAndCompute(img_src_2, mask_2)

    img_anno_1 = cv2.drawKeypoints(
        img_src_1, keypoints_1, None, (255, 0, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    img_anno_2 = cv2.drawKeypoints(
        img_src_2, keypoints_2, None, (255, 0, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    fig_kp, (ax_kp_a, ax_kp_b) = plt.subplots(
        1, 2, num="Keypoints", sharey=True, figsize=(14, 6))

    ax_kp_a.imshow(img_anno_1)
    ax_kp_b.imshow(img_anno_2)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=64)  # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(desc_1, desc_2, k=2, compactResult=True)

    # create mask to exclude vague matches
    mask_matches = np.zeros((len(matches), 2), dtype=np.uint8)

    # do ratio test
    for (match_primary, match_secondary) in matches:
        if match_primary.distance < 0.7 * match_secondary.distance:
            i = match_primary.queryIdx
            mask_matches[i, 0] = 1

    img_matches = cv2.drawMatchesKnn(img_src_1, keypoints_1, img_src_2, keypoints_2,
                                     matches, None,  matchesMask=mask_matches,
                                     flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    plt.figure("SIFT Matches", figsize=(14, 6))
    plt.imshow(img_matches)

    plt.show()
