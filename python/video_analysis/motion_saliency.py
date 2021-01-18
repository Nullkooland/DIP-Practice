import cv2
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    cap = cv2.VideoCapture(1)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    saliency = None

    while True:
        read_ok, frame = cap.read()

        if not read_ok or cv2.waitKey(16) == ord('q'):
            break

        frame = cv2.resize(frame, None, None, fx=0.5, fy=0.5)

        if saliency == None:
            saliency = cv2.saliency.MotionSaliencyBinWangApr2014_create()
            saliency.setImagesize(frame.shape[1], frame.shape[0])
            saliency.init()
        
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, saliency_map = saliency.computeSaliency(frame_gray)
        saliency_map = np.uint8(saliency_map * 255.0)

        cv2.imshow('Frame', frame)
        cv2.imshow('Saliency', saliency_map)
