import cv2
import pyheif
import numpy as np
import matplotlib.pyplot as plt

src_img = pyheif.read_as_numpy("./images/dog.heic")
src_img = cv2.resize(src_img, None, fx=0.5, fy=0.5,
                     interpolation=cv2.INTER_AREA)
# src_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)
h, w = src_img.shape[:2]

mode = 0
is_mouse_down = False
marker_colors = [
    (100, 0, 100),
    (0, 45, 255),
    (200, 0, 50),
    (0, 150, 200),
]

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
            mask = cv2.circle(mask, (x, y), 10, mode, -1, lineType=cv2.LINE_AA)
            anno_img = cv2.circle(anno_img, (x, y), 10,
                                  marker_colors[mode], -1, lineType=cv2.LINE_AA)

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


anno_img = src_img.copy()
mask = np.zeros((h, w), dtype=np.uint8)
bgd = np.zeros((1, 65), dtype=np.float64)
fgd = np.zeros((1, 65), dtype=np.float64)

cv2.namedWindow("Marker", cv2.WINDOW_KEEPRATIO)
cv2.setMouseCallback("Marker", on_mouse, (src_img, anno_img, mask, bgd, fgd))

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
        print(f"Mode is {mode} now")

cv2.destroyAllWindows()
