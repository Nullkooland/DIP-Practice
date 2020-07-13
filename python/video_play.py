import numpy as np
import cv2

if __name__ == "__main__":
    cap = cv2.VideoCapture('/Users/goose_bomb/Movies/rideon_x264.mp4')

    # don't you ever stop!!!
    cap.set(cv2.CAP_PROP_POS_FRAMES, 8726)

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # edges = cv2.Canny(frame, 50, 150)
        cv2.imshow('Ride On!!!', frame)

        if cv2.waitKey(16) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
