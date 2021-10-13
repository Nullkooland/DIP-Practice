import cv2
import numpy as np
from utils.image_reader import ImageReader


def noop(value):
    pass


if __name__ == "__main__":
    reader = ImageReader()
    src_img = reader.read("images/mountain.heic")
    src_img = cv2.cvtColor(src_img, cv2.COLOR_RGB2GRAY)
    h, w = src_img.shape

    A = src_img.astype(np.float32) / 255.0

    u, s, vt = np.linalg.svd(A, full_matrices=False)

    cv2.imshow("SVD Compression", A)
    cv2.createTrackbar("preserved singular values",
                       "SVD Compression", s.shape[0], s.shape[0], noop)

    while True:
        num_s = cv2.getTrackbarPos(
            "preserved singular values", "SVD Compression")

        s_ = np.copy(s)
        s_[num_s:] = 0

        A_svd_reconstruct = u * s_[..., None, :] @ vt
        s_ = np.int32(s_ / np.max(s_[1:]) * h)

        for i in range(num_s):
            cv2.line(A_svd_reconstruct,
                     (i, h - s_[i] - 1), (i, h - 1), (127, 127, 127, 255))

        for i in range(num_s, s.shape[0]):
            cv2.line(A_svd_reconstruct,
                     (i, h - s_[i] - 1), (i, h - 1), (127, 127, 127, 75))

        cv2.imshow("SVD Compression",  A_svd_reconstruct)

        if cv2.waitKey(33) & 0xFF == ord('q'):
            break

    print("我好了")
    cv2.destroyAllWindows()
