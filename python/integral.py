import cv2
import numpy as np
import matplotlib.pyplot as plt

src_img = cv2.imread('./images/tranquility.png')
gray_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
gray_img = ~gray_img
# gray_img = cv2.Canny(gray_img, 500, 1400, apertureSize=5)

# gray_img = np.float32(gray_img) / 255

intergal_imgs = cv2.integral3(gray_img)

plt.figure('Integral Image', figsize=(16, 5))

plt.subplot(1, 4, 1)
plt.imshow(gray_img, cmap='gray')
plt.title('Original')

plt.subplot(1, 4, 2)
plt.imshow(intergal_imgs[0], cmap='hot')
plt.title('Integral')

plt.subplot(1, 4, 3)
plt.imshow(intergal_imgs[1], cmap='hot')
plt.title('Integral - Square')


plt.subplot(1, 4, 4)
plt.imshow(intergal_imgs[2], cmap='hot')
plt.title('Integral - Tilted')

plt.tight_layout()
plt.show()