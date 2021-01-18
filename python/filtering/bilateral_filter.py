import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np

src_img = cv2.imread('./images/hanpe_tiger.png')
src_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)
src_img = cv2.normalize(src_img, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)

plt.figure('Source Image')
plt.imshow(src_img)
plt.axis('off')

fig = plt.figure('Bilateral Filter', figsize=(12, 8))
grid = ImageGrid(
    fig,
    111,  # similar to subplot(111)
    nrows_ncols=(2, 2),  # creates 2x2 grid of axes
    axes_pad=0.2,  # pad between axes in inch.
)

filtered_img = cv2.bilateralFilter(src_img, -1, 0.5, 1.0)
grid[0].imshow(filtered_img)

filtered_img = cv2.bilateralFilter(src_img, -1, 3.0, 1.0)
grid[1].imshow(filtered_img)

filtered_img = cv2.bilateralFilter(src_img, -1, 0.5, 5.0)
grid[2].imshow(filtered_img)

filtered_img = cv2.bilateralFilter(src_img, -1, 3.0, 5.0)
grid[3].imshow(filtered_img)

plt.show()
