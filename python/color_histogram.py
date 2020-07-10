import cv2
import numpy as np
import matplotlib.pyplot as plt

src_img = cv2.imread('./images/street.png')
h, w = src_img.shape[:2]

rgb_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)

hist_r = cv2.calcHist([rgb_img], [0], None, [256], [0, 256])
hist_g = cv2.calcHist([rgb_img], [1], None, [256], [0, 256])
hist_b = cv2.calcHist([rgb_img], [2], None, [256], [0, 256])

hist_r = cv2.normalize(hist_r, 0, 1, norm_type=cv2.NORM_L1)
hist_g = cv2.normalize(hist_g, 0, 1, norm_type=cv2.NORM_L1)
hist_b = cv2.normalize(hist_b, 0, 1, norm_type=cv2.NORM_L1)

hsv_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV_FULL)

hist_h = cv2.calcHist([hsv_img], [0], None, [256], [0, 256])
hist_s = cv2.calcHist([hsv_img], [1], None, [256], [0, 256])
hist_v = cv2.calcHist([hsv_img], [2], None, [256], [0, 256])

hist_h = cv2.normalize(hist_h, 0, 1, norm_type=cv2.NORM_L1)
hist_s = cv2.normalize(hist_s, 0, 1, norm_type=cv2.NORM_L1)
hist_v = cv2.normalize(hist_v, 0, 1, norm_type=cv2.NORM_L1)

plt.figure('RGB Histogram', figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(rgb_img)
plt.title('Original')

plt.subplot(1, 3, 2)
x = np.arange(256)
plt.plot(x, hist_r, color='r')
plt.plot(x, hist_g, color='g')
plt.plot(x, hist_b, color='b')
plt.legend(['R', 'G', 'B'])
plt.xlim([0, 256])
plt.title('RGB Histogram')

plt.subplot(1, 3, 3)
x = np.arange(256)
plt.plot(x, hist_h, color='c')
plt.plot(x, hist_s, color='m')
plt.plot(x, hist_v, color='y')
plt.legend(['H', 'S', 'V'])
plt.xlim([0, 256])
plt.title('HSV Histogram')

plt.tight_layout()
plt.show()

print('Histogram Comparison between R & G')
r = cv2.compareHist(hist_r, hist_g, cv2.HISTCMP_CORREL)
print(f'Correlation: {r:.4f}')
r = cv2.compareHist(hist_r, hist_g, cv2.HISTCMP_INTERSECT)
print(f'Intersection: {r:.4f}')
r = cv2.compareHist(hist_r, hist_g, cv2.HISTCMP_CHISQR_ALT)
print(f'Chi-square: {r:.4f}')
r = cv2.compareHist(hist_r, hist_g, cv2.HISTCMP_BHATTACHARYYA)
print(f'Bhattacharyya: {r:.4f}')

def hist2sig(hist):
    locs = np.arange(256, dtype=np.float32).reshape(256, 1)
    return np.concatenate((hist, locs), axis=1)


sig_r = hist2sig(hist_r)
sig_g = hist2sig(hist_g)

emd, lower_bound, flow = cv2.EMD(sig_r, sig_g, distType=cv2.DIST_L2, lowerBound=0)
print(emd)
