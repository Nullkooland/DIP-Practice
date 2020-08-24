import cv2
import numpy as np
import matplotlib.pyplot as plt

NUM_FEATURES = 2048

if __name__ == "__main__":
    # read two images of the same scene
    img_src_a = cv2.imread('./images/sleepy_cat_scale_1.png')
    img_src_b = cv2.imread('./images/sleepy_cat_scale_2.png')
    mask_a = cv2.imread(
        './images/sleepy_cat_mask_scale_1.png', cv2.IMREAD_GRAYSCALE)
    mask_b = cv2.imread(
        './images/sleepy_cat_mask_scale_2.png', cv2.IMREAD_GRAYSCALE)

    # extract features using SIFT
    sift = cv2.SIFT_create(nfeatures=NUM_FEATURES)

    keypoints_a, desc_a = sift.detectAndCompute(img_src_a, mask_a)
    keypoints_b, desc_b = sift.detectAndCompute(img_src_b, mask_b)
    img_src_a = cv2.cvtColor(img_src_a, cv2.COLOR_BGR2RGB)
    img_src_b = cv2.cvtColor(img_src_b, cv2.COLOR_BGR2RGB)

    img_anno_a = cv2.drawKeypoints(
        img_src_a, keypoints_a, None, (255, 0, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    img_anno_b = cv2.drawKeypoints(
        img_src_b, keypoints_b, None, (255, 0, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    fig_kp, (ax_kp_a, ax_kp_b) = plt.subplots(
        1, 2, num='Keypoints', sharey=True, figsize=(14, 6))

    ax_kp_a.imshow(img_anno_a)
    ax_kp_b.imshow(img_anno_b)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=64)   # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(desc_a, desc_b, k=2)

    # create mask to exclude vague matches
    mask_matches = np.zeros((len(matches), 2), dtype=np.uint8)

    # do ratio test
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.7 * n.distance:
            mask_matches[i, 0] = 1

    img_matches = cv2.drawMatchesKnn(img_src_a, keypoints_a, img_src_b, keypoints_b,
                                     matches, None, (0, 255, 0), (255, 0, 0),
                                     mask_matches,
                                     flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    plt.figure('Matches', figsize=(14, 6))
    plt.imshow(img_matches)

    plt.show()
