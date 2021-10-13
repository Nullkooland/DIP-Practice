import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils.image_reader import ImageReader

MARKERS = [
    cv2.GC_FGD,
    cv2.GC_PR_FGD,
    cv2.GC_BGD,
    cv2.GC_PR_BGD,
]

MARKER_COLORS = [
    (0, 255, 0),
    (50, 125, 50),
    (0, 0, 255),
    (50, 50, 125),
]

MODE_STR = [
    "FOREGROUND",
    "PROBABLE_FOREGROUND",
    "BACKGROUND",
    "PROBABLE_BACKGROUND",
    "FOREGROUND_RECT"
]

mode = 0
is_mouse_down = False

rect_p0 = (0, 0)
rect_p1 = (0, 0)


def on_mouse(event, x, y, flags, param):
    global mode
    global is_mouse_down
    global rect_p0
    global rect_p1

    src_img, anno_img, mask, bgd, fgd = param

    if event == cv2.EVENT_LBUTTONDOWN:
        if mode == 5:
            if is_mouse_down:
                mask[...] = cv2.GC_BGD
                mask = cv2.rectangle(mask, rect_p0, (x, y),
                                     cv2.GC_PR_FGD, -1, lineType=cv2.LINE_4)
            else:
                rect_p0 = (x, y)

        is_mouse_down = not is_mouse_down

    elif event == cv2.EVENT_MOUSEMOVE and is_mouse_down:
        if mode == 5:
            rect_p1 = (x, y)
        else:
            mask = cv2.circle(mask, (x, y), 10,
                              MARKERS[mode], -1, lineType=cv2.LINE_AA)
            anno_img = cv2.circle(anno_img, (x, y), 10,
                                  MARKER_COLORS[mode], -1, lineType=cv2.LINE_AA)

    if event == cv2.EVENT_RBUTTONDOWN:
        mask, bgd, fgd = cv2.grabCut(src_img, mask, None, bgd, fgd, 5,
                                     mode=cv2.GC_INIT_WITH_MASK)

        foreground_mask = np.logical_or(
            mask == cv2.GC_PR_FGD, mask == cv2.GC_FGD).astype(np.uint8)

        cropped_img = cv2.copyTo(src_img, foreground_mask)
        cv2.imshow("cropped", cropped_img)
        # cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)

        # plt.figure("Grabcut", figsize=(12, 4))
        # plt.subplot(1, 2, 1)
        # plt.imshow(mask, cmap="gray")
        # plt.title("Original")

        # plt.subplot(1, 2, 2)
        # plt.imshow(cropped_img)
        # plt.title("Foreground")

        # plt.subplot(1, 3, 3)
        # plt.imshow(fgd, cmap="gray")
        # plt.title("Morphological Processed Mask")

        # plt.tight_layout()
        # plt.show()


if __name__ == "__main__":
    reader = ImageReader()
    src_img = reader.read("images/dog.heic", swapRB=True)
    src_img = cv2.resize(src_img, None, fx=0.5, fy=0.5,
                         interpolation=cv2.INTER_AREA)
    h, w = src_img.shape[:2]

    anno_img = src_img.copy()
    mask = np.zeros((h, w), dtype=np.uint8)
    bgd = np.zeros((1, 65), dtype=np.float64)
    fgd = np.zeros((1, 65), dtype=np.float64)

    cv2.namedWindow("Marker", cv2.WINDOW_KEEPRATIO)
    cv2.setMouseCallback("Marker", on_mouse,
                         (src_img, anno_img, mask, bgd, fgd))

    while True:
        if mode == 5:
            cv2.copyTo(src_img, None, dst=anno_img)
            cv2.rectangle(anno_img, rect_p0, rect_p1,
                          (255, 0, 0), 2, lineType=cv2.LINE_4)

        cv2.imshow("Marker", anno_img)
        cv2.imshow("Mask", np.uint8(mask * 75))

        key = cv2.waitKey(1)

        if key == ord('q'):
            break
        if key != -1:
            mode = key - ord('0')
            print(f"Mode is [{MODE_STR[mode]}] now")

    cv2.destroyAllWindows()
