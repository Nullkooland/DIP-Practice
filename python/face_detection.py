import numpy as np
import cv2

FEATURES_FILES_DIR = '/usr/local/share/opencv4/haarcascades/'
FACE_FEATURES_FILE = 'haarcascade_frontalface_alt.xml'
EYE_FEATURES_FILE = 'haarcascade_eye.xml'
SMILE_FEATURES_FILE = 'haarcascade_smile.xml'

# Prepare this shitty 720P camera on my MBP
cap = cv2.VideoCapture(1)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f'Video input size:{w}x{h}')

# Prepare the detectors
face_detector = cv2.CascadeClassifier(FEATURES_FILES_DIR + FACE_FEATURES_FILE)
eye_detector = cv2.CascadeClassifier(FEATURES_FILES_DIR + EYE_FEATURES_FILE)
smile_detector = cv2.CascadeClassifier(
    FEATURES_FILES_DIR + SMILE_FEATURES_FILE)

background = cv2.imread('./images/loquat_painting.png')
bg_height, bg_width, _ = background.shape

tick = cv2.TickMeter()

while (True):

    # Capture one frame
    ret, frame = cap.read()
    if not ret:
        continue

    # frame = cv2.resize(frame, (1920, 1080),
    #                    interpolation=cv2.INTER_AREA)
    # frame = cv2.UMat(frame)

    tick.reset()
    tick.start()
    # Convert color to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 1.1)

    # Get detected face regions
    faces = face_detector.detectMultiScale(gray,
                                           scaleFactor=1.2,
                                           minNeighbors=5,
                                           flags=cv2.CASCADE_FIND_BIGGEST_OBJECT,
                                           minSize=(100, 100), maxSize=(600, 600))

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    tick.stop()
    fps = 1.0 / tick.getTimeSec()

    cv2.putText(frame, f'FPS: {fps:.2f}', (10, 20),
                cv2.FONT_HERSHEY_PLAIN, 1.25, (0, 255, 0), 1, lineType=cv2.LINE_AA)

    # Show frame
    cv2.imshow('Face Detection', frame)
    # cv2.imshow('Blending', blend)

    # Wait for exit
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# We're done
cap.release()
cv2.destroyAllWindows()
print('我好了')
