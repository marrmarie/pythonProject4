import cv2
import autopy
import mediapipe as mp
import math
import time
import numpy as np
import pyautogui
def cl(result):
    x8 = result.multi_hand_landmarks[0].landmark[8].x
    y8 = result.multi_hand_landmarks[0].landmark[8].y
    x4 = result.multi_hand_landmarks[0].landmark[4].x
    y4 = result.multi_hand_landmarks[0].landmark[4].y
    s48 = math.hypot(x8 - x4, y8 - y4)



cap = cv2.VideoCapture(0)
width, height = autopy.screen.size()

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
            if id == 8:
                cv2.circle(img, (cx, cy), 25, (255, 0, 255), cv2.FILLED)
                autopy.mouse.move(cx * width / w, cy * height / h)
                if cl(result):
                    pyautogui.click()
                    print(4)
                    time.sleep(3)
        mpDraw.draw_landmarks(img, result.multi_hand_landmarks[0], mp.solutions.hands.HAND_CONNECTIONS)
    #img = np.fliplr(img)
    cv2.imshow('Handtrack', img)
    cv2.waitKey(1)




