import numpy as np
import cv2

if __name__ == "__main__":
    cap = cv2.VideoCapture("./videos/freeway720.264")
    # bg_subtractor = cv2.bgsegm.createBackgroundSubtractorGMG()
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True, history=128)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    while True:
        is_read_ok, frame = cap.read()
        if not is_read_ok or cv2.waitKey(16) & 0xFF == ord('q'):
            break

        mask_fg = bg_subtractor.apply(frame)
        mask_fg = cv2.morphologyEx(mask_fg, cv2.MORPH_OPEN, kernel)
        cv2.imshow("frame", frame)
        cv2.imshow("mask", mask_fg)

    cap.release()
    cv2.destroyAllWindows()
