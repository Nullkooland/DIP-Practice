import numpy as np
import cv2
import matplotlib.pyplot as plt

# load source image
src_img = cv2.imread('./images/spike.png')
src_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)
src_img = np.float32(src_img) / 255.0

zero_kernel = np.zeros((3, 3))
sobel_kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
sobel_kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

# compute 3D-convoltion
(r, g, b) = cv2.split(src_img)
filtered_r = cv2.filter2D(r, cv2.CV_32F, sobel_kernel_y)
filtered_g = cv2.filter2D(g, cv2.CV_32F, zero_kernel)
filtered_b = cv2.filter2D(b, cv2.CV_32F, sobel_kernel_x)
filtered_3d = filtered_r + filtered_g + filtered_b

# plot result
plt.subplot(1, 2, 1)
plt.title('Source Image')
plt.imshow(src_img)

plt.subplot(1, 2, 2)
plt.title('3D Convolution Result')
plt.imshow(filtered_3d, cmap='gray')

plt.show()