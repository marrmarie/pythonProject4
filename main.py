import cv2
import autopy
import mediapipe as mp
import math
import numpy as np
import pyautogui


def cl(result):
    x8 = result.multi_hand_landmarks[0].landmark[8].x
    y8 = result.multi_hand_landmarks[0].landmark[8].y
    x12 = result.multi_hand_landmarks[0].landmark[12].x
    y12 = result.multi_hand_landmarks[0].landmark[12].y
    s128 = math.hypot(x8 - x12, y8 - y12)
    if s128 < 0.05:
        return True
    else:
        return False


def finger2(results):
    a5x = results.multi_hand_landmarks[0].landmark[5].x
    a5y = results.multi_hand_landmarks[0].landmark[5].y
    a6x = results.multi_hand_landmarks[0].landmark[6].x
    a6y = results.multi_hand_landmarks[0].landmark[6].y
    a7x = results.multi_hand_landmarks[0].landmark[7].x
    a7y = results.multi_hand_landmarks[0].landmark[7].y
    a8x = results.multi_hand_landmarks[0].landmark[8].x
    a8y = results.multi_hand_landmarks[0].landmark[8].y
    a0x = results.multi_hand_landmarks[0].landmark[0].x
    a0y = results.multi_hand_landmarks[0].landmark[0].y
    if math.hypot(a5x - a0x, a5y - a0y) < math.hypot(a6x - a0x, a6y - a0y) < math.hypot(a7x - a0x,
                                                                                        a7y - a0y) < math.hypot(a8x - a0x, a8y - a0y):
        return True
    else:
        return False


def finger3(results):
    a5x = results.multi_hand_landmarks[0].landmark[13].x
    a5y = results.multi_hand_landmarks[0].landmark[13].y
    a6x = results.multi_hand_landmarks[0].landmark[14].x
    a6y = results.multi_hand_landmarks[0].landmark[14].y
    a7x = results.multi_hand_landmarks[0].landmark[15].x
    a7y = results.multi_hand_landmarks[0].landmark[15].y
    a8x = results.multi_hand_landmarks[0].landmark[16].x
    a8y = results.multi_hand_landmarks[0].landmark[16].y
    a0x = results.multi_hand_landmarks[0].landmark[0].x
    a0y = results.multi_hand_landmarks[0].landmark[0].y
    if math.hypot(a5x - a0x, a5y - a0y) < math.hypot(a6x - a0x, a6y - a0y) < math.hypot(a7x - a0x,
                                                                                        a7y - a0y) < math.hypot(a8x - a0x, a8y - a0y):
        return True
    else:
        return False


def finger4(results):
    a5x = results.multi_hand_landmarks[0].landmark[17].x
    a5y = results.multi_hand_landmarks[0].landmark[17].y
    a6x = results.multi_hand_landmarks[0].landmark[18].x
    a6y = results.multi_hand_landmarks[0].landmark[18].y
    a7x = results.multi_hand_landmarks[0].landmark[19].x
    a7y = results.multi_hand_landmarks[0].landmark[19].y
    a8x = results.multi_hand_landmarks[0].landmark[20].x
    a8y = results.multi_hand_landmarks[0].landmark[20].y
    a0x = results.multi_hand_landmarks[0].landmark[0].x
    a0y = results.multi_hand_landmarks[0].landmark[0].y
    if math.hypot(a5x - a0x, a5y - a0y) < math.hypot(a6x - a0x, a6y - a0y) < math.hypot(a7x - a0x,
                                                                                        a7y - a0y) < math.hypot(a8x - a0x, a8y - a0y):
        return True
    else:
        return False


cap = cv2.VideoCapture(0)
width, height = autopy.screen.size()

hands = mp.solutions.hands.Hands(static_image_mode=False,
                                 max_num_hands=1,
                                 min_tracking_confidence=0.3,
                                 min_detection_confidence=0.3)
ans = 0
mpDraw = mp.solutions.drawing_utils
while True:
    _, img = cap.read()
    result = hands.process(img)
    if result.multi_hand_landmarks:
        for id, lm in enumerate(result.multi_hand_landmarks[0].landmark):
            h, w, _ = img.shape
            cx, cy = int(lm.x * w), int(lm.y * h)

            cv2.circle(img, (cx, cy), 3, (355, 0, 255))
            if finger2(result) and id == 8:
                cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
                if cx > 15 and cx < width - 180 and cy > 0 and cy < height:
                    autopy.mouse.move((width - (cx * width) / w), (cy * height) / (h))

                if cl(result) and finger2(result) and finger3(result) and finger4(result):
                    pyautogui.scroll(-5)

                elif cl(result) and finger2(result) and finger4(result):
                    pyautogui.scroll(5)

                elif cl(result) and finger2(result):
                    autopy.mouse.click()

        mpDraw.draw_landmarks(img, result.multi_hand_landmarks[0], mp.solutions.hands.HAND_CONNECTIONS)
    img = np.fliplr(img)
    cv2.imshow('Handtrack', img)
    cv2.waitKey(1)
