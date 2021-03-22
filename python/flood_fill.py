import cv2
import pyheif
import numpy as np
import matplotlib.pyplot as plt

# flood fill
LOW_DIFF = 75
HIGH_DIFF = 75
src_img = pyheif.read_as_numpy("./images/moonbear.heic")
src_img = cv2.resize(src_img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
h, w = src_img.shape[:2]

mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
ret, img, mask, rect = cv2.floodFill(src_img, mask,
                                     (384, 640),
                                     (255, 0, 0),
                                     (LOW_DIFF, LOW_DIFF, LOW_DIFF),
                                     (HIGH_DIFF, HIGH_DIFF, HIGH_DIFF),
                                     flags=cv2.FLOODFILL_MASK_ONLY | cv2.FLOODFILL_FIXED_RANGE | 4)

mask = mask[1:h+1, 1:w+1]
cropped_img = cv2.copyTo(src_img, mask)

plt.figure("Flood Fill", figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(src_img)
plt.title("Original")

plt.subplot(1, 3, 2)
plt.imshow(mask, cmap="gray")
plt.title("Mask")

plt.subplot(1, 3, 3)
plt.imshow(cropped_img)
plt.title("Cropped")

plt.tight_layout()
plt.show()

# watershed
is_mouse_down = False
marker_index = 1
marker_color = (0, 0, 255)

def on_mouse(event, x, y, flags, param):
    global is_mouse_down
    global marker_index
    global marker_color

    src_img, draw_img, markers = param
    if event == cv2.EVENT_LBUTTONDOWN:
        is_mouse_down = True
        
    elif event == cv2.EVENT_LBUTTONUP:
        is_mouse_down = False
        marker_index += 1
        marker_color = (np.random.randint(0, 256),
                        np.random.randint(0, 256),
                        np.random.randint(0, 256))

    elif event == cv2.EVENT_MOUSEMOVE and is_mouse_down:
        cv2.circle(markers, (x, y), 4, marker_index, -1, cv2.LINE_8)
        cv2.circle(draw_img, (x, y), 4, marker_color, -1, cv2.LINE_8)

    elif event == cv2.EVENT_RBUTTONDOWN:
        markers = cv2.watershed(src_img, markers)
        mask = np.uint8(markers == 1)
        cropped_img = cv2.copyTo(src_img, mask)

        plt.figure("Watershed", figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.imshow(cv2.cvtColor(draw_img, cv2.COLOR_BGR2RGB))
        plt.title("Input Markers")

        plt.subplot(1, 3, 2)
        plt.imshow(markers)
        plt.title("Semantic Map")

        plt.subplot(1, 3, 3)
        plt.imshow(cropped_img)
        plt.title("That\'s My Bear!")

        plt.tight_layout()
        plt.show()


draw_img = cv2.cvtColor(src_img, cv2.COLOR_RGB2BGR)
markers = np.zeros((h, w), dtype=np.int32)

cv2.namedWindow("Watershed")
cv2.setMouseCallback("Watershed", on_mouse, param=(src_img, draw_img, markers))

while True:
    cv2.imshow("Watershed", draw_img)
    if cv2.waitKey(16) & 0xFF == ord('q'):
        break