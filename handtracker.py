# Open-Source Code from: https://github.com/Sousannah/hand-tracking-using-mediapipe

import cv2
import mediapipe as mp
import time
from angleCalculator import calculate_angle, initialize_kalman, apply_kalman_filter

mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.5) # Limits to 1 hand for faster processing
mpDraw = mp.solutions.drawing_utils

kalman_filters = {
    'thumb': [initialize_kalman(), initialize_kalman()],
    'index': [initialize_kalman(), initialize_kalman()],
    'middle': [initialize_kalman(), initialize_kalman()],
    'ring': [initialize_kalman(), initialize_kalman()],
    'pinky': [initialize_kalman(), initialize_kalman()]
}

def get_joint_angles(landmarks, finger_points, kalman_filters):
    """Calculate and smooth angles for a finger's joints."""
    angles = []
    for i in range(len(finger_points) - 2):
        a, b, c = finger_points[i], finger_points[i+1], finger_points[i+2]
        angle = calculate_angle(a, b, c)
        smoothed_angle = apply_kalman_filter(kalman_filters[i], angle)
        angles.append(smoothed_angle)
    return angles

def draw_angles_on_image(img, angles, joints, color=(255, 0, 0)):
    """Draw angles on the image at specified joint positions."""
    for angle, joint in zip(angles, joints):
        cv2.putText(img, f'{int(angle)}°', joint, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

def main():
    cap = cv2.VideoCapture(0)
    pTime = 0

    while True:
        success, img = cap.read()
        if not success:
            break

        img = cv2.flip(img, 1)
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resizes image for faster processing
        small_img = cv2.resize(imgRGB, (320, 240))
        results = hands.process(small_img)

        if results and results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                landmarks = [(lm.x, lm.y, lm.z) for lm in handLms.landmark]
                h, w, c = img.shape

                # Define finger landmarks (Base, Middle 1, Middle 2, Tip)
                finger_points = {
                    'thumb': [landmarks[1], landmarks[2], landmarks[3], landmarks[4]],
                    'index': [landmarks[5], landmarks[6], landmarks[7], landmarks[8]],
                    'middle': [landmarks[9], landmarks[10], landmarks[11], landmarks[12]],
                    'ring': [landmarks[13], landmarks[14], landmarks[15], landmarks[16]],
                    'pinky': [landmarks[17], landmarks[18], landmarks[19], landmarks[20]]
                }

                # Calculate and smooth angles for each finger
                for finger, points in finger_points.items():
                    angles = get_joint_angles(landmarks, points, kalman_filters[finger])
                    joints = [(int(p[0] * w), int(p[1] * h)) for p in points[1:3]]
                    draw_angles_on_image(img, angles, joints)

                # Draw landmarks
                mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

        # Calculate and display FPS
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        cv2.imshow("Hand Tracker", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()