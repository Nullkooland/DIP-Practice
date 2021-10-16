import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils.image_reader import ImageReader

# KERNEL_RADIUS = 300

if __name__ == "__main__":
    reader = ImageReader()
    src_img = reader.read("images/rmb_coins.heic")
    template = reader.read("images/one_yuan_coin.heic")
    template = cv2.resize(template, (192, 192))

    h, w = src_img.shape[:2]
    th, tw = template.shape[:2]

    matching = cv2.matchTemplate(src_img, template, cv2.TM_CCOEFF_NORMED)
    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(matching)

    cv2.rectangle(src_img, maxLoc,
                  (maxLoc[0] + tw, maxLoc[1] + th), (255, 0, 0), 2)
    cv2.rectangle(matching, maxLoc, (maxLoc[0] + tw, maxLoc[1] + th), -1, 2)

    plt.figure("Template Matching", figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(src_img)

    plt.subplot(1, 2, 2)
    plt.imshow(matching, cmap="jet", vmin=-1, vmax=1)

    plt.tight_layout()
    plt.show()

    template = reader.read("images/eye_template.heic")
    cv2.imshow("Head Template", template)

    template = cv2.resize(template, None, fx=0.4, fy=0.4)
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    th, tw = template.shape[:2]

    cap = cv2.VideoCapture(1)
    while True:
        _, frame = cap.read()
        frame = cv2.resize(frame, (640, 360))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        match = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)

        _, maxVal, _, maxLoc = cv2.minMaxLoc(match)
        cv2.rectangle(
            frame, maxLoc, (maxLoc[0] + tw, maxLoc[1] + th), (255, 0, 0), 2)

        match_disp = np.uint8((match + 1) / 2 * 255)
        match_disp = cv2.applyColorMap(match_disp, cv2.COLORMAP_JET)
        cv2.drawMarker(match_disp, maxLoc, (0, 0, 0), cv2.MARKER_CROSS, 10, 2)

        cv2.imshow("Video", frame)
        cv2.imshow("Template Matching", match_disp)

        if cv2.waitKey(16) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
