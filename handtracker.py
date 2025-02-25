# Open-Source Code from: https://github.com/Sousannah/hand-tracking-using-mediapipe

import cv2
import mediapipe as mp
import time
from angleCalculator import calculate_angle, kalman_filter, thumb_kf, index_kf, middle_kf, ring_kf, pinky_kf

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            # Get (x, y, z) positions of landmarks
            landmarks = [(lm.x, lm.y, lm.z) for lm in handLms.landmark]

            # Define finger landmarks (Base, Middle, Tip)
            thumb_points = [landmarks[1], landmarks[2], landmarks[4]]
            index_points = [landmarks[5], landmarks[6], landmarks[8]]
            middle_points = [landmarks[9], landmarks[10], landmarks[12]]
            ring_points = [landmarks[13], landmarks[14], landmarks[16]]
            pinky_points = [landmarks[17], landmarks[18], landmarks[20]]

            # Calculate angles using your function
            thumb_angle = kalman_filter(thumb_kf, calculate_angle(*thumb_points))
            index_angle = kalman_filter(index_kf, calculate_angle(*index_points))
            middle_angle = kalman_filter(middle_kf, calculate_angle(*middle_points))
            ring_angle = kalman_filter(ring_kf, calculate_angle(*ring_points))
            pinky_angle = kalman_filter(pinky_kf, calculate_angle(*pinky_points))

            # Convert normalized (x, y) values to image pixel positions
            h, w, c = img.shape
            n = 2 # 1 = base, 2 = middle, 4 = tip
            thumb_joint = (int(landmarks[n][0] * w), int(landmarks[n][1] * h))
            index_joint = (int(landmarks[n+4][0] * w), int(landmarks[n+4][1] * h))
            middle_joint = (int(landmarks[n+8][0] * w), int(landmarks[n+8][1] * h))
            ring_joint = (int(landmarks[n+12][0] * w), int(landmarks[n+12][1] * h))
            pinky_joint = (int(landmarks[n+16][0] * w), int(landmarks[n+16][1] * h))

            # Display angles on screen
            cv2.putText(img, f'{int(thumb_angle)}°', thumb_joint, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            cv2.putText(img, f'{int(index_angle)}°', index_joint, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            cv2.putText(img, f'{int(middle_angle)}°', middle_joint, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            cv2.putText(img, f'{int(ring_angle)}°', ring_joint, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            cv2.putText(img, f'{int(pinky_angle)}°', pinky_joint, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            # Draw landmarks
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                (255, 0, 255), 3)

    cv2.imshow("Hand Tracker", img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()