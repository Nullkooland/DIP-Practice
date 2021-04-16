import cv2
import numpy as np
import matplotlib.cm as colormap
from typing import List, Tuple


def on_mouse_event(event, x, y, flags, param):
    list_tracking_points: List[Tuple(int, int)] = param

    if event == cv2.EVENT_LBUTTONDOWN:
        list_tracking_points.append((x, y))
    elif event == cv2.EVENT_RBUTTONDOWN:
        list_tracking_points.clear()


def get_tracking_points(cap: cv2.VideoCapture) -> np.ndarray:
    cv2.namedWindow("Specify tracking points")

    tracking_points = None
    list_tracking_points = []
    cv2.setMouseCallback("Specify tracking points", on_mouse_event,
                         param=list_tracking_points)

    while True:
        is_read_ok, frame = cap.read()
        if not is_read_ok:
            break

        if cv2.waitKey(16) == ord('e'):
            # Refine to get sub-pixel corners
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            criteria = (cv2.TermCriteria_COUNT +
                        cv2.TermCriteria_EPS, 32, 1e-2)
            tracking_points = np.stack(list_tracking_points).reshape(
                (-1, 1, 2)).astype(np.float32)
            tracking_points = cv2.cornerSubPix(
                frame_gray, tracking_points, (7, 7), (-1, -1), criteria)
            break

        for point in list_tracking_points:
            cv2.circle(frame, point, 4, (150, 0, 150), -1)

        cv2.imshow("Specify tracking points", frame)

    cv2.destroyWindow("Specify tracking points")
    return tracking_points


if __name__ == "__main__":
    cap = cv2.VideoCapture(1)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    frame = np.empty((h, w, 3), dtype=np.uint8)
    prev_frame = np.empty_like(frame)

    frame_gray = np.empty((h, w), dtype=np.uint8)
    prev_frame_gray = np.empty_like(frame_gray)

    # Specify tracking points manually
    tracking_points_prev = get_tracking_points(cap)
    num_tracking_points = tracking_points_prev.shape[0]

    # Read first frame
    cap.read(image=prev_frame)
    cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY, prev_frame_gray)

    # Get colors
    cm = colormap.get_cmap('tab10')
    colors = cm(range(num_tracking_points), alpha=False, bytes=True)

    while True:
        is_read_ok, _ = cap.read(image=frame)
        cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY, dst=frame_gray)

        if not is_read_ok or (cv2.waitKey(16) == ord('q')):
            break

        tracking_points, status, err = cv2.calcOpticalFlowPyrLK(
            prev_frame_gray, frame_gray, tracking_points_prev, None)

        mask = (status == 1).flatten()
        tracking_points = tracking_points[mask, ...]

        for i, point in enumerate(tracking_points):
            x = int(point[:, 0])
            y = int(point[:, 1])
            color = colors[i].astype(np.float64)
            cv2.circle(frame, (x, y), 6, color, -1, cv2.LINE_AA)

        cv2.imshow("pre", prev_frame_gray)
        cv2.imshow("Sparse optical flow", frame)

        if len(tracking_points) == 0:
            print("All tracking points lost!")
            break

        tracking_points_prev = tracking_points
        # cv2.copyTo(frame, None, dst=prev_frame)
        cv2.copyTo(frame_gray, None, dst=prev_frame_gray)

    cv2.destroyAllWindows()
