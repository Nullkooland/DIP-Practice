import cv2
import numpy as np

if __name__ == "__main__":
    dis = cv2.DISOpticalFlow.create(cv2.DISOpticalFlow_PRESET_MEDIUM)
    cap = cv2.VideoCapture(1)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    prev_frame_gray = np.zeros((h, w), dtype=np.uint8)
    flow_hsv = np.full((h, w, 3), 255, dtype=np.uint8)

    while True:
        is_read_ok, frame = cap.read()
        
        if not is_read_ok or (cv2.waitKey(16) & 0xFF == ord('q')):
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = dis.calc(prev_frame_gray, frame_gray, None)
        # flow = cv2.calcOpticalFlowFarneback(prev_frame_gray, frame_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        prev_frame_gray = frame_gray

        mag_flow, angle_flow = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        flow_hsv[..., 0] = cv2.normalize(angle_flow, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        flow_hsv[..., 2] = cv2.normalize(mag_flow, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        flow_bgr = cv2.cvtColor(flow_hsv, cv2.COLOR_HSV2BGR_FULL)

        cv2.imshow("Frame", frame)
        cv2.imshow("Optical flow", flow_bgr)
        

    cap.release()
    cv2.destroyAllWindows()