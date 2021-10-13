import cv2
import numpy as np

BUFFER_SIZE = 3

if __name__ == "__main__":
    cap = cv2.VideoCapture(1)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) / 2)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) / 2)

    tick = cv2.TickMeter()

    frame_buffer = []
    curr_i = 0

    while True:
        tick.reset()
        tick.start()

        ret, frame = cap.read()
        frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)

        if len(frame_buffer) < BUFFER_SIZE:
            frame_buffer.append(frame)
            continue
        else:
            frame_buffer[curr_i] = frame
            curr_i = (curr_i + 1) % BUFFER_SIZE

        # denoised = cv2.fastNlMeansDenoisingColoredMulti(
        #     frame_buffer, 1, BUFFER_SIZE, h=7, hColor=10, templateWindowSize=3, searchWindowSize=11)

        denoised = cv2.fastNlMeansDenoisingColored(
            frame_buffer[curr_i], h=7, hColor=10, templateWindowSize=3, searchWindowSize=13)

        tick.stop()
        fps = 1.0 / tick.getTimeSec()

        cv2.imshow("Original", frame_buffer[curr_i])

        cv2.putText(denoised, f"FPS: {fps:.2f}", (10, 20),
                    cv2.FONT_HERSHEY_PLAIN, 1.25, (0, 255, 0), 1, lineType=cv2.LINE_AA)
        cv2.imshow("Denoised", denoised)

        curr_i = (curr_i + 1) % BUFFER_SIZE

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
