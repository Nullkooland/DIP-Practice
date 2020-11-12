import numpy as np
import cv2

if __name__ == "__main__":
    cap = cv2.VideoCapture(1)
    fgbg = cv2.createBackgroundSubtractorMOG2()
    

    while True:
        ret, frame = cap.read()

        fgmask = fgbg.apply(frame)
        cv2.imshow('frame', fgmask)

        if cv2.waitKey(30) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()