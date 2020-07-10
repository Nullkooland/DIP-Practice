import cv2
import numpy as np
import matplotlib.pyplot as plt

target_img = cv2.imread('./images/human_skin.png')
cv2.imshow('Target', target_img)

src_img = cv2.imread('./images/trek.png')
cv2.imshow('Image to Match', src_img)

cv2.waitKey()

target_lab = cv2.cvtColor(target_img, cv2.COLOR_BGR2LAB)

hist_l = cv2.calcHist([target_lab], [0], None, [256], [0, 256])
hist_a = cv2.calcHist([target_lab], [1], None, [256], [0, 256])
hist_b = cv2.calcHist([target_lab], [2], None, [256], [0, 256])

hist_ab = cv2.calcHist([target_lab], [1, 2], None, [5, 5], [128, 160, 128, 160])
# hist_ab = cv2.normalize(hist_ab, 0, 1, norm_type=cv2.NORM_L1)

plt.figure('Lab Histogram', figsize=(10, 4))

plt.subplot(1, 2, 1)
x = np.arange(256)
plt.plot(x, hist_l, color='black')
plt.legend('L')

plt.subplot(1, 2, 2)
x = np.arange(0, 36)
plt.plot(x, hist_a[128:164], color='y')
plt.plot(x, hist_b[128:164], color='c')
plt.legend(['A', 'B'])
plt.xlim([0, 36])

plt.tight_layout()
plt.show()

src_lab = cv2.cvtColor(src_img, cv2.COLOR_BGR2LAB)
match = cv2.calcBackProject([src_lab], [1, 2], hist_ab, [128, 160, 128, 160], scale=1)

match = cv2.applyColorMap(match, cv2.COLORMAP_HOT)
cv2.imshow('Back Project', match)
cv2.waitKey()


cap = cv2.VideoCapture(0)
# frame_buffer = np.empty((3, 720, 1280, 3), dtype=np.uint8)
# i = 0
pre_match = np.zeros((720, 1280), dtype=np.uint8)

while True:
    _, frame = cap.read()
    frame = cv2.pyrDown(frame)
    # frame = cv2.medianBlur(frame, 7)

    cv2.imshow('Frame', frame)

    frame_lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    match = cv2.calcBackProject([frame_lab], [1, 2], hist_ab, [128, 160, 128, 160], scale=1)

    # match |= pre_match
    # pre_match = match

    mask = cv2.applyColorMap(match, cv2.COLORMAP_BONE)
    mask[match < 8] = 0
    cv2.imshow('Back Project', mask)

    if cv2.waitKey(16) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()