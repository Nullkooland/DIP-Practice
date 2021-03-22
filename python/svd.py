import numpy as np
import cv2
import pyheif


def noop(value):
    pass


src_img = pyheif.read_as_numpy("./images/mountain.heic")
h, w = src_img.shape

A = src_img.astype(np.float32) / 255.0

u, s, vt = np.linalg.svd(A, full_matrices=False)

cv2.imshow("SVD Compression", A)
cv2.createTrackbar("preserved singular values",
                   "SVD Compression", s.shape[0], s.shape[0], noop)


while True:
    num_s = cv2.getTrackbarPos("preserved singular values", "SVD Compression")

    s_ = np.copy(s)
    s_[num_s:] = 0

    A_svd_reconstruct = u * s_[..., None,:] @ vt
    s_ = np.int32(s_ / np.max(s_[1:]) * h)
    
    for i in range(num_s):
        cv2.line(A_svd_reconstruct, (i, h - s_[i] - 1), (i, h - 1), (127, 127, 127, 255))

    for i in range(num_s, s.shape[0]):
        cv2.line(A_svd_reconstruct, (i, h - s_[i] - 1), (i, h - 1), (127, 127, 127, 75))

    cv2.imshow("SVD Compression",  A_svd_reconstruct)

    if cv2.waitKey(33) & 0xFF == ord('q'):
        break

print("我好了")
cv2.destroyAllWindows()
