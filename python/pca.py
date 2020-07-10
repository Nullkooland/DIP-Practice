import cv2
import numpy as np

src_img = cv2.imread('./images/rice.png', cv2.IMREAD_GRAYSCALE)
cv2.imshow('rice', src_img)

# binary threshold to get mask
ret, mask = cv2.threshold(src_img, 25, 255, cv2.THRESH_BINARY)
# mask = cv2.adaptiveThreshold(src_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 9, 3)

# apply morphological filters to remove noises
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

cv2.imshow('mask', mask)

# get contours
contours, hierarchy = cv2.findContours(
    mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)

img = cv2.cvtColor(src_img, cv2.COLOR_GRAY2BGR)
img = cv2.drawContours(img, contours, -1, (0, 0, 255), 1, lineType=cv2.LINE_AA)

# compute PCA
for contour in contours:
    contour = np.squeeze(contour, axis=1).astype(np.float32)
    center, directions, eigenvalues = cv2.PCACompute2(contour, np.empty(0))
    # draw direction vectors for each rice seed
    x, y = center[0]
    v0 = directions[0]
    v1 = directions[1]

    len0 = np.dot(contour - center, v0)
    len1 = np.dot(contour - center, v1)

    v0 *= np.max(len0)
    v1 *= np.max(len1)

    cv2.circle(img, (x, y), 7, (0, 0, 255), 1, lineType=cv2.LINE_AA)

    cv2.arrowedLine(img, (x, y), (x + v0[0], y + v0[1]),
                    (0, 255, 0), 2, tipLength=0.25, line_type=cv2.LINE_AA)

    cv2.arrowedLine(img, (x, y), (x + v1[0], y + v1[1]),
                    (200, 75, 0), 2, tipLength=0.5, line_type=cv2.LINE_AA)


cv2.imshow('rice contours', img)

cv2.waitKey()
