import cv2
import mediapipe as mp
import numpy as np
import mouse
from screeninfo import get_monitors

# Initializing the Model
mp_hands = mp.solutions.hands
HandLandmark = mp_hands.HandLandmark

hands = mp_hands.Hands(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.75,
    max_num_hands=1,
)

# Start capturing video from webcam
cap = cv2.VideoCapture(0)

monitors = get_monitors()
SCREENSIZE = np.array([monitors[0].height, monitors[0].width], dtype="int")


def main():

    # Read video frame by frame
    success, img = cap.read()

    # Flip the image(frame)
    img = cv2.flip(img, 1)

    # Convert BGR image to RGB image
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Process the RGB image
    results = hands.process(imgRGB)

    # If hands are present in image(frame)
    if results.multi_hand_landmarks:
        landmarks = results.multi_hand_landmarks[0].landmark
        index_pos = get_landmark_position(landmarks, HandLandmark.INDEX_FINGER_TIP)
        thumb_pos = get_landmark_position(landmarks, HandLandmark.THUMB_TIP)

        mouse_pos = ((index_pos + thumb_pos) / 2 * SCREENSIZE).astype(int)
        mouse.move(mouse_pos[0], mouse_pos[1], absolute=True)

        finger_vect = np.absolute(index_pos - thumb_pos)
        finger_gap = np.sum(finger_vect)

    cv2.imshow("Image", img)


def get_landmark_position(landmarks, finger):
    return np.array([landmarks[finger].x, landmarks[finger].y], dtype=float)


while True:
    main()
    # Display Video and when 'q' is entered, destroy the window
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
