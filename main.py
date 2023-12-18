import cv2
import autopy
import mediapipe as mp
import math
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
def finger2(result):
    x8 = result.multi_hand_landmarks[0].landmark[8].x
    y8 = result.multi_hand_landmarks[0].landmark[8].y
    x5 = result.multi_hand_landmarks[0].landmark[5].x
    y5 = result.multi_hand_landmarks[0].landmark[5].y
    x0 = result.multi_hand_landmarks[0].landmark[0].x
    y0 = result.multi_hand_landmarks[0].landmark[0].y
    s80 = math.hypot(x8 - x0, y8 - y0)
    s50 = math.hypot(x5 - x0, y5 - y0)
    if s50 < s80:
        return True
    else:
        return False



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
            if finger2(result) and id == 8:
                cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
                autopy.mouse.move((cx * width) / w, (cy * height) / (h + 300))

                if cl(result) and finger2(result):
                    autopy.mouse.click()
                    print(4)
                    #time.sleep(3)
        mpDraw.draw_landmarks(img, result.multi_hand_landmarks[0], mp.solutions.hands.HAND_CONNECTIONS)
    #img = np.fliplr(img)
    cv2.imshow('Handtrack', img)
    cv2.waitKey(1)



