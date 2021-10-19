import cv2
import numpy as np
import io
from matplotlib import colors
from matplotlib import cm

MODEL_DIR = "./dnn_models/yolo"
VARIANT = "yolov5s"

CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.5
NUM_COLORS = 16
INPUT_SIZE = (640, 640)
IS_INPUT_SIZE_SQUARE = (INPUT_SIZE[0] == INPUT_SIZE[1])

if __name__ == "__main__":
    # Load YOLOv5 ONNX model
    model = cv2.dnn.readNetFromONNX(f"{MODEL_DIR}/{VARIANT}.onnx")
    output_layers = model.getUnconnectedOutLayersNames()

    # Load class names
    with io.open('./dnn_models/yolo/coco.names', 'r') as f:
        names = f.read().splitlines()

    # Prepare camera
    cap = cv2.VideoCapture(1)
    ih = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    iw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    s = min(ih, iw)

    if IS_INPUT_SIZE_SQUARE:
        box_scale = np.array(
            [s / INPUT_SIZE[0], s / INPUT_SIZE[1], s / INPUT_SIZE[0], s / INPUT_SIZE[1]])
    else:
        box_scale = np.array(
            [iw / INPUT_SIZE[0], ih / INPUT_SIZE[1], iw / INPUT_SIZE[0], ih / INPUT_SIZE[1]])

    # Prepare colors for different class
    cmap = cm.get_cmap('tab10')
    colors = cmap(range(NUM_COLORS), bytes=True)

    # Start capturing
    while True:
        ret, frame = cap.read()
        # YOLOv5 is sensitive to aspect ratio, so we just crop center here
        if IS_INPUT_SIZE_SQUARE:
            frame = frame[ih // 2 - s // 2: ih // 2 + s // 2,
                          iw // 2 - s // 2: iw // 2 + s // 2, :]

        if not ret or cv2.waitKey(1) == ord('q'):
            break

        blob = cv2.dnn.blobFromImage(
            frame, 1.0 / 255.0, INPUT_SIZE, swapRB=True, crop=False)

        # Input frame
        model.setInput(blob)

        # Run inference and measure time spent
        start_time = cv2.getTickCount()
        outputs = model.forward(output_layers)
        end_time = cv2.getTickCount()
        infer_time = (end_time - start_time) / cv2.getTickFrequency()

        detections = outputs[0].squeeze()

        # Filter out predictions with low confidence
        mask = detections[..., 4] > CONFIDENCE_THRESHOLD
        detections = detections[mask]

        # Compute confidence: conf = obj_conf * class_conf
        detections[..., 5:] *= detections[..., 4:5]

        # Get class ids
        class_ids = np.argmax(detections[..., 5:], axis=1)
        confidences = np.take_along_axis(
            detections[..., 5:], np.expand_dims(class_ids, axis=1), axis=1).squeeze()

        # Extract boxes
        boxes = detections[..., :4] * np.expand_dims(box_scale, axis=0)
        # [c_x, c_y, w, h] -> [l_x, t_y, w, h]
        boxes[..., :2] -= 0.5 * boxes[..., 2:4]

        indices = cv2.dnn.NMSBoxes(
            boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)

        if len(indices) == 0:
            cv2.imshow(VARIANT, frame)
            continue
        else:
            # Draw all detected boxes
            for i in indices:
                confidence = confidences[i]

                box = boxes[i]
                x = int(box[0])
                y = int(box[1])
                w = int(box[2])
                h = int(box[3])

                class_id = class_ids[i]
                color = colors[class_id % NUM_COLORS]
                b = int(color[2])
                g = int(color[1])
                r = int(color[0])

                name = names[class_id]

                cv2.rectangle(frame, (x, y), (x + w, y + h),
                              (b, g, r), 2, lineType=cv2.LINE_4)

                cv2.putText(frame, f'{name}: {confidence * 100:.1f}%', (x, y - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (b, g, r), 1, lineType=cv2.LINE_AA)

            cv2.putText(frame, f'FPS: {1.0/infer_time:.2f}', (32, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, lineType=cv2.LINE_AA)
            cv2.imshow(VARIANT, frame)

    cv2.destroyAllWindows()
