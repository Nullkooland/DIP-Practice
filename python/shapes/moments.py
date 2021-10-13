import cv2
import numpy as np
from utils.image_reader import ImageReader
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # load image with alpha channel
    reader = ImageReader()
    img_raw = reader.read("images/opencv_logo.heic", ignore_alpha=False)
    img_src = img_raw[..., :3].copy()
    mask = img_raw[..., 3]
    # load shape template
    img_template = reader.read("images/opencv_logo_part.heic")
    img_template = cv2.cvtColor(img_template, cv2.COLOR_RGB2GRAY)
    # set the transparent background as white
    img_bg = cv2.bitwise_or(img_src, (255, 255, 255), mask=~mask)
    img_src += img_bg

    # get the contour in shape template
    _, img_template_mask = cv2.threshold(
        img_template, 75, 255, cv2.THRESH_BINARY_INV)
    contour_template, _ = cv2.findContours(
        img_template_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
    contour_template = contour_template[0]
    moments_template = cv2.moments(contour_template)
    hu_template = cv2.HuMoments(moments_template)

    # find all connected components in target image
    contours, hierarchy = cv2.findContours(
        mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_L1)

    # draw contours
    cv2.drawContours(img_src, contours, -1, (255, 0, 255),
                     2, cv2.LINE_AA, hierarchy)

    # match contours found with template contour
    for i, contour in enumerate(contours):
        moments = cv2.moments(contour, True)
        hu = cv2.HuMoments(moments)

        area = moments["m00"]
        x = moments["m10"] / area
        y = moments["m01"] / area

        a = moments["m20"] - moments["m10"] ** 2 / moments["m00"]
        c = moments["m02"] - moments["m01"] ** 2 / moments["m00"]
        b = moments["m11"] - moments["m10"] * moments["m01"] / moments["m00"]
        orientation = 0.5 * cv2.fastAtan2(2 * b, a - c)

        x_arrow_s = int(x)
        y_arrow_s = int(y)
        x_arrow_e1 = int(x + 50 * np.cos(np.deg2rad(orientation)))
        y_arrow_e1 = int(y + 50 * np.sin(np.deg2rad(orientation)))
        x_arrow_e2 = int(x + 50 * np.cos(np.deg2rad(orientation + 90)))
        y_arrow_e2 = int(y + 50 * np.sin(np.deg2rad(orientation + 90)))
        cv2.arrowedLine(img_src, (x_arrow_s, y_arrow_s), (x_arrow_e1, y_arrow_e1),
                        (0, 255, 150), 1, line_type=cv2.LINE_AA)
        cv2.arrowedLine(img_src, (x_arrow_s, y_arrow_s), (x_arrow_e2, y_arrow_e2),
                        (0, 150, 255), 1, line_type=cv2.LINE_AA)

        print(f"Shape [{i}]")
        print(
            f"Area: {area:.1f} Center: ({x:.1f}, {y:.1f}) Orientation: {orientation:.1f}")
        cv2.putText(img_src, f"{i}", (int(x), int(y)),
                    cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.5, (255, 0, 255), 1, lineType=cv2.LINE_AA)

        print("Hu moments:")
        print(hu)

        match_score = cv2.matchShapes(
            contour, contour_template, cv2.CONTOURS_MATCH_I3, 0)
        print(f"Match score: {match_score:.4f}\n")

    print(f"Template")
    print("Hu moments:")
    print(hu_template)
    print("\n")

    plt.figure("Original", figsize=(6, 6))
    plt.imshow(img_src)

    plt.figure("Shape Template", figsize=(2, 2))
    plt.imshow(img_template)
    plt.show()
