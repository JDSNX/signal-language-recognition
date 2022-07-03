import mediapipe as mp
import cv2

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = holistic.process(image)
        
        res_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Right Hand Landmarks
        mp_drawing.draw_landmarks(res_image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        
        # Left Hand Landmarks
        mp_drawing.draw_landmarks(res_image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        cv2.imshow('Hollistic Model Detection', res_image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()