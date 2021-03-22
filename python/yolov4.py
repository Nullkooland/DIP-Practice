import cv2
import numpy as np
import io
from matplotlib import colors
from matplotlib import cm

MODEL_DIR = "./dnn_models/yolo/"
CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.6
MAX_NUM_DETECTIONS = 64
NUM_COLORS = 16

if __name__ == "__main__":
    # Load YOLOv4 model
    model = cv2.dnn_DetectionModel(
        MODEL_DIR + "yolov4.cfg", MODEL_DIR + "yolov4.weights")
    
    # # Load YOLOv4 tiny model
    # model = cv2.dnn_DetectionModel(
    #     MODEL_DIR + "yolov4_tiny.cfg", MODEL_DIR + "yolov4_tiny.weights")

    output_layers = model.getUnconnectedOutLayersNames()

    # Load class names
    with io.open("./dnn_models/yolo/coco.names", "r") as f:
        names = f.read().splitlines()

    # Prepare camera
    cap = cv2.VideoCapture(0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    box_scale = np.array([w, h, w, h])

    num_detection = 0
    boxes = []
    confidences = np.empty(MAX_NUM_DETECTIONS, dtype=np.float)
    class_ids = np.empty(MAX_NUM_DETECTIONS, dtype=np.int)

    # Prepare colors for different class
    cmap = cm.get_cmap("tab10")
    colors = cmap(range(NUM_COLORS), bytes=True)

    # Start capturing
    while True:
        ret, frame = cap.read()

        if not ret or cv2.waitKey(16) == ord('q'):
            break

        blob = cv2.dnn.blobFromImage(
            frame, 1.0 / 255.0, (416, 416), swapRB=True, crop=False)

        # Input frame
        model.setInput(blob)

        # Run inference and measure time spent
        start_time = cv2.getTickCount()
        outputs = model.forward(output_layers)
        end_time = cv2.getTickCount()
        infer_time = (end_time - start_time) / cv2.getTickFrequency()

        num_detection = 0
        boxes.clear()

        # Extract from each output layer
        for output in outputs:
            # Extract each detection
            for detection in output:
                # Find which class the object with
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                # Reject the detection with low confidence
                if confidence < CONFIDENCE_THRESHOLD:
                    continue

                box = (detection[:4] * box_scale).astype(np.int)
                # convert from center to top-left corner
                box[:2] -= box[2:] // 2

                # Store the intermediary results
                boxes.append(box)
                confidences[num_detection] = confidence
                class_ids[num_detection] = class_id
                num_detection += 1

        # Draw FPS
        cv2.putText(frame, f"FPS: {1.0/infer_time:.2f}", (32, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, lineType=cv2.LINE_AA)

        if num_detection == 0:
            cv2.imshow("YOLO v4", frame)
            continue

        # Do Non-maximum Suppression
        indices = cv2.dnn.NMSBoxes(boxes, confidences[:num_detection],
                                   CONFIDENCE_THRESHOLD, NMS_THRESHOLD)

        # Draw all detected boxes
        for i in indices.flatten():
            confidence = confidences[i]

            box = boxes[i]
            x = box[0]
            y = box[1]
            width = box[2]
            height = box[3]

            class_id = class_ids[i]
            color = colors[class_id % NUM_COLORS]
            b = int(color[2])
            g = int(color[1])
            r = int(color[0])

            name = names[class_id]
            
            cv2.rectangle(frame, (x, y), (x + width, y + height),
                          (b, g, r), 2, lineType=cv2.LINE_4)

            cv2.putText(frame, f"{name}: {confidence * 100:.1f}%", (x, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (b, g, r), 1, lineType=cv2.LINE_AA)

        cv2.imshow("YOLO v4", frame)

    cv2.destroyAllWindows()
