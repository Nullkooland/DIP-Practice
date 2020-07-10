import numpy as np
import cv2
import matplotlib.pyplot as plt

def cartoonize(img):
    num_down = 2        #缩减像素采样的数目
    num_bilateral = 3    #定义双边滤波的数目
    #用高斯金字塔降低取样
    img_color = img
    for _ in range(num_down):
        img_color = cv2.pyrDown(img_color)
    #重复使用小的双边滤波代替一个大的滤波
    for _ in range(num_bilateral):
        img_color = cv2.bilateralFilter(img_color,d=9,sigmaColor=9,sigmaSpace=7)
    #升采样图片到原始大小
    for _ in range(num_down):
        img_color = cv2.pyrUp(img_color)
    #转换为灰度并且使其产生中等的模糊
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_blur = cv2.medianBlur(img_gray, 7)
    #检测到边缘并且增强其效果
    img_edge = cv2.adaptiveThreshold(img_blur,255,
                                     cv2.ADAPTIVE_THRESH_MEAN_C,
                                     cv2.THRESH_BINARY,
                                     blockSize=9,
                                     C=2)
    #转换回彩色图像
    img_edge = cv2.cvtColor(img_edge, cv2.COLOR_GRAY2RGB)
    img_cartoon = cv2.bitwise_and(img_color, img_edge)
    return img_cartoon

src_img = cv2.imread('./images/racoon.png')
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
filtered_img = cv2.sepFilter2D(src_img, cv2.CV_32F, kernel_l, gaussian_kernel)

plt.subplot(1, 2, 1)
plt.title('Source Image')
plt.imshow(src_img)

plt.subplot(1, 2, 2)
plt.title('Filtered Image')
plt.imshow(filtered_img)

plt.show()
