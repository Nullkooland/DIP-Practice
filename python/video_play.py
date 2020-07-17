import numpy as np
import cv2
import struct

DONT_YOU_EVER_STOP_BEGIN = 11402
DONT_YOU_EVER_STOP_END = 11714

if __name__ == "__main__":
    cap = cv2.VideoCapture('/Users/goose_bomb/Movies/rideon_x264.mp4')

    # don't you ever stop!!!
    cap.set(cv2.CAP_PROP_POS_FRAMES, 5023)
    # cap.set(cv2.CAP_PROP_POS_FRAMES, 5120)
    M = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    N = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc_vals = bytearray(struct.pack('i', int(cap.get(cv2.CAP_PROP_FOURCC))))
    print([ f'{chr(c)}' for c in fourcc_vals ])

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'H264')
    out = cv2.VideoWriter('/Users/goose_bomb/Movies/rideon_polar.mp4', fourcc, 30, (N, M), True)

    assert out.isOpened()
    # frame_pos = DONT_YOU_EVER_STOP_BEGIN

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # edges = cv2.Canny(frame, 50, 150)
        polar_img = cv2.logPolar(
            frame, (N // 2, M // 2), 274, cv2.WARP_FILL_OUTLIERS | cv2.INTER_CUBIC)

        cv2.imshow('Ride On!!!', polar_img)
        out.write(polar_img)

        # frame_pos += 1
        # if frame_pos == DONT_YOU_EVER_STOP_END:
        #     break

        if cv2.waitKey(16) == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
