import numpy as np
import cv2

FEATURE_DESC_FILE = '/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_alt.xml'

# Prepare this shitty 720P camera on my MBP
cap = cv2.VideoCapture(1)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f'Video input size:{w}x{h}')

# Prepare the detector
detector = cv2.CascadeClassifier(FEATURE_DESC_FILE)

background = cv2.imread('./images/loquat_painting.png')
bg_height, bg_width, _ = background.shape

tick = cv2.TickMeter()

while (True):

    # Capture one frame
    ret, frame = cap.read()
    # frame = cv2.resize(cv2.UMat(frame), (640, 360),
    #                    interpolation=cv2.INTER_AREA)
    # frame = cv2.UMat(frame)
    tick.reset()
    tick.start()
    # Convert color to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 1.1)

    # Get detected face regions
    faces = detector.detectMultiScale(gray,
                                      scaleFactor=1.08,
                                      minNeighbors=5,
                                      flags=cv2.CASCADE_SCALE_IMAGE,
                                      minSize=(100, 100),
                                      maxSize=(600, 600))

    for (x, y, w, h) in faces:
        h_ex = h // 4
        h_ex = min(h_ex, y)
        h_roi = min(h + h_ex * 2, bg_height)
        w_roi = min(w, bg_width)

        # roi = frame[y - h_ex:y - h_ex + h_roi, x:x + w_roi]
        # pos = (bg_width // 2, h_roi // 2)

        # blend = cv2.seamlessClone(roi, background, None, pos, cv2.MIXED_CLONE)
        frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    tick.stop()
    fps = 1.0 / tick.getTimeSec()

    cv2.putText(frame, f'FPS: {fps:.2f}', (10, 20),
                cv2.FONT_HERSHEY_PLAIN, 1.25, (0, 255, 0), 1, lineType=cv2.LINE_AA)

    # Show frame
    cv2.imshow('Face Detection', frame)
    # cv2.imshow('Blending', blend)

    # Wait for exit
    if cv2.waitKey(16) & 0xFF == ord('q'):
        break

# We're done
cap.release()
cv2.destroyAllWindows()
print('我好了')
