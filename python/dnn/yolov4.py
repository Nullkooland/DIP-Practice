import cv2
import numpy as np
import io
from matplotlib import colors
from matplotlib import cm

MODEL_DIR = "./dnn_models/yolo"
VARIANT = "yolov4_tiny"

CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.6
NUM_COLORS = 16

if __name__ == "__main__":
    # Load YOLOv4 tiny model
    net = cv2.dnn.readNet(f"{MODEL_DIR}/{VARIANT}.cfg",
                          f"{MODEL_DIR}/{VARIANT}.weights")
    model = cv2.dnn_DetectionModel(net)
    model.setInputParams(size=(416, 416), scale=1/255, swapRB=True, crop=False)

    # Load class names
    with io.open("./dnn_models/yolo/coco.names", "r") as f:
        names = f.read().splitlines()

    # Prepare video input
    cap = cv2.VideoCapture(1)

    # Prepare colors for different class
    cmap = cm.get_cmap("tab10")
    colors = cmap(range(NUM_COLORS), bytes=True)
    colors = cv2.cvtColor(colors.reshape(-1, 1, 4), cv2.COLOR_RGBA2BGR)

    # Start capturing
    while True:
        is_read_ok, frame = cap.read()

        if not is_read_ok or cv2.waitKey(16) == ord('q'):
            break

        # Run inference and measure time spent
        start_time = cv2.getTickCount()
        classes, scores, boxes = model.detect(
            frame, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
        end_time = cv2.getTickCount()
        infer_time = (end_time - start_time) / cv2.getTickFrequency()

        for (class_id, score, box) in zip(classes, scores, boxes):
            color = colors[class_id % NUM_COLORS].flatten().tolist()
            label = names[int(class_id)]

            cv2.rectangle(frame, box, color, 2)
            cv2.putText(frame, label, (box[0], box[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        # Draw FPS
        cv2.putText(frame, f"FPS: {1.0/infer_time:.2f}", (32, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, lineType=cv2.LINE_AA)

        cv2.imshow("YOLO v4", frame)

    cv2.destroyAllWindows()
