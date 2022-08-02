import mediapipe as mp
import cv2
import os
import numpy as np
import csv

from pathlib import Path

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

DATA_PATH = os.path.join('mediapipe_data1')
KW_PATH = r'D:\keywords.csv'

no_of_videos = 30 
frames_of_video = 30

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def prob_viz(res, actions, input_frame):  
    colors = [(245,117,16), (117,245,16), (16,117,245)]
    input_frame = cv2.resize(input_frame, dsize=None, fx=1.4, fy=1.4)
    input_frame = cv2.flip(input_frame, 1)
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85+num*40), cv2.
        FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
    return output_frame

def get_size(action):
    size = 0
    for path, dirs, files in os.walk(os.path.join(DATA_PATH, action)):
        for f in files:
            fp = os.path.join(path, f)
            size += os.path.getsize(fp)
    return size

def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(132)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(63)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(63)
    return np.concatenate([pose, face, lh, rh])

def get_keywords():
    create_csv()
    with open(KW_PATH) as f:
        reader = csv.reader(f)
        return np.array([col for row in reader for col in row])

def create_csv():
    if not os.path.exists(KW_PATH):
        fle = Path(KW_PATH)
        fle.touch(exist_ok=True)
        f = open(fle)
        return False
    return True

def create_mp_data():
    for action in get_keywords(): 
        for video in range(no_of_videos):
            try:
                os.makedirs(os.path.join(DATA_PATH, action, str(video)))
            except:
                pass

def init():
    create_csv()
    create_mp_data()
    