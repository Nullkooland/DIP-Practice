import cv2
import pyheif
import numpy as np

MODEL_DIR = "./dnn_models/retinaface/"
MODEL_PROFILE = "mnet025_v2"
SCRORE_THRESHOLD = 0.95
NMS_THRESHOLD = 0.6
MAX_NUM_DETECTIONS = 256
STRIDES = [32, 16, 8]

def generate_anchors(strides, image_size):
    w, h = image_size
    anchors = []
    pass

if __name__ == "__main__":
    model = cv2.dnn.readNetFromONNX(MODEL_DIR + MODEL_PROFILE + ".onnx")
    output_layers = model.getUnconnectedOutLayersNames()

    # Prepare camera
    cap = cv2.VideoCapture(1)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    anchors = generate_anchors(STRIDES, image_size)
    scale = np.array([w, h, w, h])

    detected_boxes = []
    detected_scores = np.empty(MAX_NUM_DETECTIONS, dtype=np.float)
    detected_landmarks = np.empty((MAX_NUM_DETECTIONS, 5, 2), dtype=np.float)
    num_detection = 0

    while True:
        ret, frame = cap.read()

        if not ret or cv2.waitKey(16) == ord('q'):
            break

        blob = cv2.dnn.blobFromImage(
            frame, 1, (640, 640), swapRB=True, crop=False)

        # Input frame
        model.setInput(blob)

        # Run inference and measure time spent
        start_time = cv2.getTickCount()
        outputs = model.forward(output_layers)
        

        end_time = cv2.getTickCount()
        infer_time = (end_time - start_time) / cv2.getTickFrequency()

        detected_boxes.clear()
        num_detection = 0

        for stride_level, stride in enumerate(STRIDES):
            # Extract face info at this level
            scores = outputs[stride_level * 3]
            bbox_offsets = outputs[stride_level * 3 + 1]
            landmarks = outputs[stride_level * 3 + 2]

            height, width = bbox_offsets.shape[2:]




        for i, score in enumerate(scores):
            if score[0] < SCRORE_THRESHOLD or num_detection >= MAX_NUM_DETECTIONS:
                continue

            box = boxes[i] * scale
            box[2:] -= box[:2]

            detected_boxes.append(box)
            detected_landmarks[num_detection] = np.reshape(
                landmarks[i], (5, 2))
            detected_scores[num_detection] = score[0]
            num_detection += 1

        # Draw FPS
        cv2.putText(frame, f"FPS: {1.0/infer_time:.2f}", (32, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, lineType=cv2.LINE_AA)

        if num_detection == 0:
            cv2.imshow("RetinaFace", frame)
            continue

        indices = cv2.dnn.NMSBoxes(detected_boxes, detected_scores[:num_detection],
                                   SCRORE_THRESHOLD, NMS_THRESHOLD)

        # Draw all detected faces
        for i in indices.flatten():
            score = detected_scores[i]

            box = detected_boxes[i]
            x0 = int(box[0])
            y0 = int(box[1])
            x1 = int(box[2])
            y1 = int(box[3])

            cv2.rectangle(frame, (x0, y0), (x1, y1),
                          (0, 0, 255), 2, lineType=cv2.LINE_4)

            cv2.putText(frame, f"Face: {score * 100:.1f}%", (x0, y0 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, lineType=cv2.LINE_AA)

        cv2.imshow("RetinaFace", frame)

    cv2.destroyAllWindows()
