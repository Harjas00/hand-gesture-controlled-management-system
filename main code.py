import cv2
import mediapipe as mp
from math import hypot
import screen_brightness_control as sbc
import numpy as np
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume  # for volume control
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL

# Initialize MediaPipe Hands
mpHands = mp.solutions.hands
hands = mpHands.Hands(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.75,
    max_num_hands=2  # Track both hands
)
Draw = mp.solutions.drawing_utils

# Initialize volume control
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

# Get the volume range (min, max, increment)
minVol, maxVol, _ = volume.GetVolumeRange()

# Start capturing video from webcam
cap = cv2.VideoCapture(0)

# Define thresholds for hand ranges
BRIGHTNESS_MIN_DIST, BRIGHTNESS_MAX_DIST = 15, 220
VOLUME_MIN_DIST, VOLUME_MAX_DIST = 15, 220

while True:
    # Read video frame by frame
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)  # Flip image
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR image to RGB

    # Process the RGB image
    Process = hands.process(frameRGB)
    landmarkLists = []  # List to store landmarks for both hands

    # Check if hands are present
    if Process.multi_hand_landmarks:
        for handlm in Process.multi_hand_landmarks:
            landmarks = []
            height, width, _ = frame.shape

            # Gather landmarks
            for _id, lm in enumerate(handlm.landmark):
                x, y = int(lm.x * width), int(lm.y * height)
                landmarks.append([_id, x, y])

            landmarkLists.append(landmarks)  # Append each hand's landmarks to the list
            Draw.draw_landmarks(frame, handlm, mpHands.HAND_CONNECTIONS)  # Draw hand landmarks

    # Initialize the hand variables
    leftHand = None
    rightHand = None

    # Check if one or both hands are detected
    if len(landmarkLists) == 1:
        # Only one hand detected, decide based on which hand is visible
        hand = landmarkLists[0]
        # Check the x-coordinate of the wrist (landmark 0) to decide which hand it is
        if hand[0][1] < frame.shape[1] // 2:  # Left hand (left side of screen)
            leftHand = hand
        else:  # Right hand (right side of screen)
            rightHand = hand
    elif len(landmarkLists) == 2:
        # Both hands detected, assign accordingly
        if landmarkLists[0][0][1] < landmarkLists[1][0][1]:  # Compare x-coordinates
            leftHand, rightHand = landmarkLists[0], landmarkLists[1]
        else:
            leftHand, rightHand = landmarkLists[1], landmarkLists[0]

    # Right hand controls brightness (if visible)
    if rightHand:
        x_1, y_1 = rightHand[4][1], rightHand[4][2]  # Thumb tip
        x_2, y_2 = rightHand[8][1], rightHand[8][2]  # Index finger tip
        cv2.circle(frame, (x_1, y_1), 7, (0, 255, 0), cv2.FILLED)
        cv2.circle(frame, (x_2, y_2), 7, (0, 255, 0), cv2.FILLED)
        cv2.line(frame, (x_1, y_1), (x_2, y_2), (0, 255, 0), 3)
        brightness_distance = hypot(x_2 - x_1, y_2 - y_1)
        brightness_level = np.interp(brightness_distance, [BRIGHTNESS_MIN_DIST, BRIGHTNESS_MAX_DIST], [0, 100])
        sbc.set_brightness(int(brightness_level))

    # Left hand controls volume (if visible)
    if leftHand:
        x_1, y_1 = leftHand[4][1], leftHand[4][2]  # Thumb tip
        x_2, y_2 = leftHand[8][1], leftHand[8][2]  # Index finger tip
        cv2.circle(frame, (x_1, y_1), 7, (255, 0, 0), cv2.FILLED)
        cv2.circle(frame, (x_2, y_2), 7, (255, 0, 0), cv2.FILLED)
        cv2.line(frame, (x_1, y_1), (x_2, y_2), (255, 0, 0), 3)
        volume_distance = hypot(x_2 - x_1, y_2 - y_1)

        # Adjust volume range mapping
        volume_level = np.interp(volume_distance, [VOLUME_MIN_DIST, VOLUME_MAX_DIST], [minVol, maxVol])
        volume.SetMasterVolumeLevel(volume_level, None)

    # Display video
    cv2.imshow('Image', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
