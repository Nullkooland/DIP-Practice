import numpy as np
import cv2
import matplotlib.pyplot as plt

if __name__ == "__main__":
    src_img = cv2.imread('./images/spike.png')
    src_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)
    src_img = np.float32(src_img) / 255.0

    # kernel = np.array([
    #     [-1, -1, -1],
    #     [-1, +8, -1],
    #     [-1, -1, -1]], dtype=float)
        
    # filtered_img = cartoonize(src_img)
    gaussian_kernel = cv2.getGaussianKernel(7, 2.0, ktype=cv2.CV_32F)
    kernel_l = gaussian_kernel[:4]
    kernel_l /= np.sum(kernel_l)
    kernel_r = np.flip(kernel_l, axis=0)
    print(kernel_l)
    print(kernel_r)
    filtered_img = cv2.sepFilter2D(src_img, cv2.CV_32F, kernel_l, kernel_r)

    plt.subplot(1, 2, 1)
    plt.title('Source Image')
    plt.imshow(src_img)

    plt.subplot(1, 2, 2)
    plt.title('Filtered Image')
    plt.imshow(filtered_img)

    plt.show()
