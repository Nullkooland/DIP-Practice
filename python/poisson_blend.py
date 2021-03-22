import numpy as np
import cv2
import pyheif

# Load images
foreground = pyheif.read_as_numpy("./images/polar_bear.heic")
background = pyheif.read_as_numpy("./images/loquat_painting.heic")

# foreground = cv2.resize(foreground, None, fx=0.5, fy=0.5)
mask = np.full_like(foreground, 255)

(height, width, channels) = background.shape
pos = (120, 64)

blend = cv2.seamlessClone(foreground, background, mask, pos, cv2.MIXED_CLONE)

cv2.imshow("Blend Result", blend)
cv2.waitKey()
