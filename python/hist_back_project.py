import cv2
import numpy as np
import matplotlib.pyplot as plt

AB_BINS = 24
AB_RANGE_LOW = 128
AB_RANGE_HIGH = 152

img_template = cv2.imread('./images/hand.png')
mask_template = cv2.imread('./images/hand_mask.png', cv2.IMREAD_GRAYSCALE)
img_template = cv2.bitwise_and(img_template, (255, 255, 255), mask=mask_template)

cv2.imshow('Template', img_template)

img_src = cv2.imread('./images/trek.png')
cv2.imshow('Image to Match', img_src)

cv2.waitKey()

lab_template = cv2.cvtColor(img_template, cv2.COLOR_BGR2LAB)

hist_l = cv2.calcHist([lab_template], [0], mask_template, [256], [0, 256])
hist_a = cv2.calcHist([lab_template], [1], mask_template, [256], [0, 256])
hist_b = cv2.calcHist([lab_template], [2], mask_template, [256], [0, 256])

hist_ab = cv2.calcHist([lab_template], [1, 2], None, [AB_BINS, AB_BINS], [
                       AB_RANGE_LOW, AB_RANGE_HIGH, AB_RANGE_LOW, AB_RANGE_HIGH])
# hist_ab = cv2.normalize(hist_ab, 0, 1, norm_type=cv2.NORM_L1)

plt.figure('Lab Histogram', figsize=(10, 4))

plt.subplot(1, 2, 1)
x = np.arange(256)
plt.plot(x, hist_l, color='black')
plt.legend('L')

plt.subplot(1, 2, 2)
x = np.arange(AB_RANGE_LOW, AB_RANGE_HIGH)
plt.plot(x, hist_a[AB_RANGE_LOW:AB_RANGE_HIGH], color='y')
plt.plot(x, hist_b[AB_RANGE_LOW:AB_RANGE_HIGH], color='c')
plt.legend(['A', 'B'])
plt.xlim([AB_RANGE_LOW, AB_RANGE_HIGH])

plt.tight_layout()
plt.show()

src_lab = cv2.cvtColor(img_src, cv2.COLOR_BGR2LAB)
match = cv2.calcBackProject([src_lab], [1, 2], hist_ab, [
                            AB_RANGE_LOW, AB_RANGE_HIGH, AB_RANGE_LOW, AB_RANGE_HIGH], scale=1)

match = cv2.applyColorMap(match, cv2.COLORMAP_HOT)
cv2.imshow('Back Project', match)
cv2.waitKey()


cap = cv2.VideoCapture(0)
# frame_buffer = np.empty((3, 720, AB_RANGE_LOW0, 3), dtype=np.uint8)
# i = 0
pre_match = np.zeros((720, 1280), dtype=np.uint8)

while True:
    _, frame = cap.read()
    # frame = cv2.pyrDown(frame)
    frame = cv2.medianBlur(frame, 3)

    cv2.imshow('Frame', frame)

    frame_lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    match = cv2.calcBackProject([frame_lab], [1, 2], hist_ab, [
                                AB_RANGE_LOW, AB_RANGE_HIGH, AB_RANGE_LOW, AB_RANGE_HIGH], scale=1)

    # match |= pre_match
    # pre_match = match

    mask = cv2.applyColorMap(match, cv2.COLORMAP_BONE)
    mask[match < 8] = 0
    cv2.imshow('Back Project', mask)

    if cv2.waitKey(16) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
