import cv2
import numpy as np
import matplotlib.pyplot as plt

src_img = cv2.imread('./images/corn_kernels.png', cv2.IMREAD_UNCHANGED)
h, w = src_img.shape[:2]

mask = src_img[..., 3]
src_img = cv2.cvtColor(src_img[...,:3], cv2.COLOR_BGR2RGB)
gray_img = cv2.cvtColor(src_img, cv2.COLOR_RGB2GRAY)
gray_img = cv2.equalizeHist(gray_img)
gray_img = np.repeat(gray_img[..., np.newaxis], 3, axis=2)

# distance transform
dist = cv2.distanceTransform(mask, cv2.DIST_L2, cv2.DIST_MASK_5, dstType=cv2.CV_32F)
# locate center of each kernel
thresh, dist_thresh = cv2.threshold(dist, 0.42 * np.max(dist), 255, cv2.THRESH_BINARY)
dist_thresh = np.uint8(dist_thresh)

# apply morphological filters to eliminate connections
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 7))
morph_mask = cv2.erode(mask, kernel, iterations=6)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
morph_mask = cv2.dilate(morph_mask, kernel, iterations=4)

# get CC
_, dist_markers = cv2.connectedComponents(dist_thresh)
n_cc, morph_markers = cv2.connectedComponents(morph_mask)
# annotate background as 1
dist_markers += 1
morph_markers += 1
# annotate edge area as 0
dist_edge = mask - dist_thresh
morph_edge = mask - morph_mask
dist_markers[dist_edge != 0] = 0
morph_markers[morph_edge != 0] = 0

plt.figure('Corn!', figsize=(12, 8))
sm = plt.cm.ScalarMappable(cmap='nipy_spectral')
cmap = sm.to_rgba(np.linspace(0, 1, n_cc + 1), bytes=True)[:,:3]
cmap[0, :] = 255
cmap[1, :] = 0

plt.subplot(2, 3, 1)
plt.imshow(src_img)
plt.title('Original')

plt.subplot(2, 3, 4)
plt.imshow(mask, cmap='gray')
plt.title('Original Mask')

plt.subplot(2, 3, 2)
plt.imshow(dist_markers, cmap='nipy_spectral')
plt.title('Distance Threshold')

plt.subplot(2, 3, 3)
plt.imshow(morph_markers, cmap='nipy_spectral')
plt.title('Morphological Processed Mask')

# Watershed
dist_markers = cv2.watershed(src_img, dist_markers)
morph_markers = cv2.watershed(src_img, morph_markers)

# color mapping
dist_markers_colored = np.take(cmap, dist_markers, axis=0)
morph_markers_colored = np.take(cmap, morph_markers, axis=0)

dist_markers_colored = cv2.addWeighted(gray_img, 0.4, dist_markers_colored, 0.6, 0)
morph_markers_colored = cv2.addWeighted(gray_img, 0.4, morph_markers_colored, 0.6, 0)

plt.subplot(2, 3, 5)
plt.imshow(dist_markers_colored)
plt.title('Watershed - Distance')

plt.subplot(2, 3, 6)
plt.imshow(morph_markers_colored)
plt.title('Watershed - Morph')

plt.tight_layout()
plt.show()