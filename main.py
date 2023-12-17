import cv2
import autopy
import mediapipe as mp

cap = cv2.VideoCapture(0)

hands = mp.solutions.hands.Hands(static_image_mode=False,
                         max_num_hands=1,
                         min_tracking_confidence=0.5,
                         min_detection_confidence=0.5)

mpDraw = mp.solutions.drawing_utils

while True:
    _, img = cap.read()
    result = hands.process(img)
    if result.multi_hand_landmarks:
        for id, lm in enumerate(result.multi_hand_landmarks[0].landmark):
            h, w, _ = img.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(img, (cx, cy), 3, (355, 0, 255))
        mpDraw.draw_landmarks(img, result.multi_hand_landmarks[0], mp.solutions.hands.HAND_CONNECTIONS)
    cv2.imshow('Handtrack', img)
    cv2.waitKey(1)



