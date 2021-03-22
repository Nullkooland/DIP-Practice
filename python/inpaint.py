import cv2
import pyheif
import numpy as np

WIN_NAME = "Inpaint"
is_mouse_down = False

# mouse callback function
def on_mouse(event, x, y, flags, param):
    global is_mouse_down
    src_img, mask = param
    if event == cv2.EVENT_LBUTTONDOWN:
        is_mouse_down = True
    elif event == cv2.EVENT_LBUTTONUP:
        is_mouse_down = False
        cv2.inpaint(src_img, mask, 5, cv2.INPAINT_NS, dst=src_img)
        mask[...] = 0
    elif event == cv2.EVENT_MOUSEMOVE and is_mouse_down:
        cv2.circle(src_img, (x, y), 5, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.circle(mask, (x, y), 5, 255, -1, cv2.LINE_AA)


if __name__ == "__main__":

    src_img = pyheif.read_as_numpy("./images/autumn.heic")
    mask = np.zeros(src_img.shape[:2], dtype=np.uint8)
    # cv2.imshow("image", src_img)

    cv2.namedWindow(WIN_NAME, cv2.WINDOW_KEEPRATIO)
    cv2.setMouseCallback(WIN_NAME, on_mouse, (src_img, mask))

    while(True):
        cv2.imshow(WIN_NAME, src_img)
        # cv2.imshow("Mask", mask)
        if cv2.waitKey(16) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
