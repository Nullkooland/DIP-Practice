import cv2
import depthai as dai
import numpy as np
import matplotlib.pyplot as plt
from utils.image_reader import ImageReader

HIST_BINS = 10
A_RANGE_LOW = 130
A_RANGE_HIGH = 154
B_RANGE_LOW = 130
B_RANGE_HIGH = 150

HIST_RANGE_LOW = 100
HIST_RANGE_HIGH = 180


def test_image(img_src, img_template, hist_ab):
    fig0, axs = plt.subplots(1, 3, figsize=(18, 6))

    axs[0].imshow(img_template)
    axs[0].set_title("Template")

    axs[1].imshow(img_src)
    axs[1].set_title("Image to Match")

    # np.save("output/skin_BA_hist.npy", hist_ab)
    plt.figure("Lab Histogram", figsize=(10, 4))

    plt.subplot(1, 2, 1)
    x = np.arange(256)
    plt.plot(x, hist_L, color="black")
    plt.legend("L")

    plt.subplot(1, 2, 2)
    x = np.arange(HIST_RANGE_LOW, HIST_RANGE_HIGH)
    plt.plot(x, hist_a[HIST_RANGE_LOW:HIST_RANGE_HIGH], color="magenta")
    plt.plot(x, hist_b[HIST_RANGE_LOW:HIST_RANGE_HIGH], color="yellow")
    plt.legend(["a", "b"])
    plt.xlim([HIST_RANGE_LOW, HIST_RANGE_HIGH])

    src_lab = cv2.cvtColor(img_src, cv2.COLOR_RGB2LAB)
    match = cv2.calcBackProject([src_lab], [1, 2], hist_ab, [
                                A_RANGE_LOW, A_RANGE_HIGH, B_RANGE_LOW, B_RANGE_HIGH], scale=1)

    axs[2].imshow(match, cmap="hot")
    axs[2].set_title("Match")

    plt.show()


def test_video(hist_ab):
    # Start defining a pipeline
    pipeline = dai.Pipeline()

    # Define a source - color camera
    cam_rgb = pipeline.createColorCamera()
    cam_rgb.setBoardSocket(dai.CameraBoardSocket.RGB)
    cam_rgb.setResolution(
        dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam_rgb.setInterleaved(True)
    cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)

    # Aeate output
    xout_rgb = pipeline.createXLinkOut()
    xout_rgb.setStreamName("rgb")
    cam_rgb.video.link(xout_rgb.input)

    # Pipeline defined, now the device is connected to
    with dai.Device(pipeline) as device:
        # Start pipeline
        device.startPipeline()

        # Output queue will be used to get the rgb frames from the output defined above
        q_rgb = device.getOutputQueue(name="rgb", maxSize=8, blocking=False)

        frame_bgr = np.zeros((1080, 1920, 3), dtype=np.uint8)
        frame_lab = np.zeros((1080, 1920, 3), dtype=np.uint8)
        mask = np.zeros((1080, 1920), dtype=np.uint8)
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        while True:
            in_rgb = q_rgb.get()  # blocking call, will wait until a new data has arrived

            # Retrieve 'bgr' (opencv format) frame
            frame_bgr = in_rgb.getCvFrame()

            cv2.medianBlur(frame_bgr, ksize=3, dst=frame_bgr)
            cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB, dst=frame_lab)

            match = cv2.calcBackProject([frame_lab], [1, 2], hist_ab, [
                A_RANGE_LOW, A_RANGE_HIGH, B_RANGE_LOW, B_RANGE_HIGH], scale=1)

            cv2.threshold(match, 200, 255, cv2.THRESH_BINARY, dst=mask)
            cv2.morphologyEx(mask, cv2.MORPH_OPEN, se, dst=mask)
            cv2.morphologyEx(mask, cv2.MORPH_CLOSE, se, dst=mask)

            cv2.imshow("Frame", frame_bgr)
            cv2.imshow("Back Project", mask)

            if cv2.waitKey(16) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    reader = ImageReader()
    img_src = reader.read("images/trek.heic")
    img_template = reader.read("images/hand.heic", ignore_alpha=False)
    mask_template = img_template[..., 3]
    img_template = img_template[..., :3]

    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask_template = cv2.morphologyEx(
        mask_template, cv2.MORPH_OPEN, se, iterations=3)

    img_template = cv2.bitwise_and(
        img_template, (255, 255, 255), mask=mask_template)

    lab_template = cv2.cvtColor(img_template, cv2.COLOR_RGB2LAB)

    hist_L = cv2.calcHist([lab_template], [0], mask_template, [256], [0, 256])
    hist_a = cv2.calcHist([lab_template], [1], mask_template, [256], [0, 256])
    hist_b = cv2.calcHist([lab_template], [2], mask_template, [256], [0, 256])

    hist_ab = cv2.calcHist([lab_template], [1, 2], None, [HIST_BINS, HIST_BINS], [
        A_RANGE_LOW, A_RANGE_HIGH, B_RANGE_LOW, B_RANGE_HIGH])

    test_image(img_src, img_template, hist_ab)
    test_video(hist_ab)
