import cv2
import pyheif
import numpy as np
import matplotlib.pyplot as plt

TEMPLATE_SIZE = 63
AREA_THRESHOLD = 300
SOLIDITY_THRESHOLD = 0.6
APPROX_POLY_RATIO = 0.04
DEFECT_FARPOINT_REL_DIST_THRESHOLD = 1
CENTER_REL_STD_THRESHOLD = 0.011


if __name__ == "__main__":
    img_src = pyheif.read_as_numpy("./images/cross_test_2.heic")
    img_anno = img_src.copy()

    img_gray = cv2.cvtColor(img_src, cv2.COLOR_RGB2GRAY)
    h, w = img_src.shape[:2]

    # denoise
    img_denoised = cv2.bilateralFilter(img_gray, -1, 8.0, 16.0)
    # find edges
    edges = cv2.Canny(~img_denoised, 200, 240)
    # do morphological operations to tweak edges
    structure_ele = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, structure_ele)

    # find contours (only the outmost ones)
    contours_raw, hierarchy = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)

    # # mark all shapes
    # img_anno = img_src.copy()
    # img_anno = cv2.drawContours(img_anno, contours, -1, (0, 255, 0), 1)

    # build cross template

    template = np.zeros((TEMPLATE_SIZE, TEMPLATE_SIZE), dtype=np.uint8)
    cv2.line(template, (TEMPLATE_SIZE // 2, 4),
             (TEMPLATE_SIZE // 2, TEMPLATE_SIZE - 5), 255, 2, cv2.LINE_4)
    cv2.line(template, (4, TEMPLATE_SIZE // 2),
             (TEMPLATE_SIZE - 5, TEMPLATE_SIZE // 2), 255, 2, cv2.LINE_4)

    template_contour, _ = cv2.findContours(
        template, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    template_contour = template_contour[0]

    # find crosses and mark them
    contours = []
    cross_contours = []
    for contour in contours_raw:
        area = cv2.contourArea(contour)
        arc_len = cv2.arcLength(contour, True)

        if area < AREA_THRESHOLD:
            continue
        # match_score = cv2.matchShapes(contour, template_contour, cv2.CONTOURS_MATCH_I1, 0)

        hull = cv2.convexHull(contour, returnPoints=False)
        try:
            defects = cv2.convexityDefects(contour, hull)
        except:
            continue

        if defects is None:
            continue

        hull_points = contour[np.squeeze(hull)]
        hull_area = cv2.contourArea(hull_points)
        solidity = area / float(hull_area)

        if solidity > SOLIDITY_THRESHOLD:
            continue

        # print(f"Solidity: {solidity:.2f}")
        contours.append(contour)

        if defects.shape[0] < 4:
            continue

        cross_far_points = []

        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]

            rel_dist = d / area
            if rel_dist < DEFECT_FARPOINT_REL_DIST_THRESHOLD:
                continue

            # print(f"Relative distance: {rel_dist:.2f}")

            start = tuple(contour[s][0])
            end = tuple(contour[e][0])
            far = tuple(contour[f][0])
            cv2.line(img_anno, start, end, (0, 0, 255), 2)
            cv2.circle(img_anno, far, 3, (255, 0, 0), -1)

            cross_far_points.append(far)

        if (len(cross_far_points) != 4):
            continue

        cross_far_points_std = np.std(np.array(cross_far_points), axis=0)
        cross_far_points_std /= area

        if cross_far_points_std[0] > CENTER_REL_STD_THRESHOLD or cross_far_points_std[1] > CENTER_REL_STD_THRESHOLD:
            continue

        print(cross_far_points_std)
        cross_contours.append(contour)

    cv2.drawContours(img_anno, contours, -1, (0, 255, 0), 1)
    cv2.drawContours(img_anno, cross_contours, -1, (255, 0, 255), -1)

    fig, (ax_src, ax_edges, ax_anno) = plt.subplots(
        1, 3, num="Find Cross", figsize=(12, 4))

    ax_src.imshow(img_denoised)
    ax_edges.imshow(edges)
    ax_anno.imshow(img_anno)

    ax_src.set_title("Denoised")
    ax_edges.set_title("Mask")
    ax_anno.set_title("Anno")

    plt.show()
