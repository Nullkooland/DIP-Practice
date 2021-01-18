import cv2
import numpy as np
import matplotlib.pyplot as plt

BACKGROUND_COLOR = np.array((170, 170, 170), dtype=np.uint8)

if __name__ == "__main__":
    img_src = cv2.imread('./images/shapes_on_paper.png')
    img_src = cv2.cvtColor(img_src, cv2.COLOR_BGR2RGB)

    # pre-process image to smooth out details
    img_mean_shifted = cv2.pyrMeanShiftFiltering(img_src, 20, 50, maxLevel=1)

    structure_ele = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    img_mean_shifted = cv2.morphologyEx(
        img_mean_shifted, cv2.MORPH_CLOSE, structure_ele, iterations=3)

    mask = cv2.inRange(img_mean_shifted, BACKGROUND_COLOR -
                       30, BACKGROUND_COLOR + 40)
    mask = ~mask

    num_cc, labels_cc, stats_cc, centroids_cc = \
        cv2.connectedComponentsWithStats(
            mask, connectivity=8, ltype=cv2.CV_16U)

    for i in range(1, num_cc):
        x, y, width, height, area = stats_cc[i]
        mass_center = tuple(np.int32(centroids_cc[i]))
        cv2.rectangle(img_src, (x, y), (x + width - 1,
                                        y + height - 1), (200, 0, 0), 2)
        cv2.drawMarker(img_src, mass_center, (0, 200, 255),
                       cv2.MARKER_TILTED_CROSS, 15, 3)
        cv2.putText(img_src, f'Area: {area:.0f}', (x - 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 0, 0), 2)

    # fast circle approximation without using hough transform
    x, y, width, height, area = stats_cc[4]
    roi = mask[y:y + height, x:x + width]

    contours, hierarchy = cv2.findContours(
        roi, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS, offset=(x, y))
    
    inner_circle = contours[1]
    center, radius = cv2.minEnclosingCircle(inner_circle)
    center = tuple(np.int32(center))

    cv2.circle(img_src, center, int(radius), (100, 255, 0), 2, cv2.LINE_AA)
    cv2.drawMarker(img_src, center, (50, 150, 0),
                       cv2.MARKER_CROSS, 11, 2)

    # show results
    fig, (ax_stats, ax_labels) = plt.subplots(1, 2, figsize=(12, 6))
    ax_stats.imshow(img_src)
    ax_stats.set_title('Stats')

    ax_labels.imshow(labels_cc, cmap='nipy_spectral')
    ax_labels.set_title('Labels')

    plt.show()
