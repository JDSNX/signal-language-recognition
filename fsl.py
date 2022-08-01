import cv2
import numpy as np
import os
import time
import mediapipe as mp

from module import *
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical, plot_model
from keras.models import load_model
from matplotlib import pyplot as plt

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import LSTM, Dense
from tensorflow.python.keras.callbacks import TensorBoard


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

actions = get_keywords()
no_of_videos = 30 #no_sequence
frames_of_video = 30 #sequence_length

for action in actions: 
    for video in range(no_of_videos):
        try:
            os.makedirs(os.path.join(DATA_PATH, action, str(video)))
        except:
            pass

cap = cv2.VideoCapture(0)


def training_testing():
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        for action in actions:
            if get_size(action) == 0:
                for video in range(no_of_videos):
                    for frame_num in range(frames_of_video):

                        ret, frame = cap.read()

                        image, results = mediapipe_detection(frame, holistic)
                        draw_landmarks(image, results)

                        if frame_num == 0: 
                            cv2.putText(image, 'STARTING COLLECTION', (120,200), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 3, cv2.LINE_AA)
                            cv2.putText(image, 'Video No.: {}'.format(video), (15,12), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

                            cv2.imshow(str(action).capitalize(), image)
                            cv2.waitKey(1000)
                        else: 
                            cv2.putText(image, 'Video No.: {}'.format(video), (15,12), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                            cv2.imshow(str(action).capitalize(), image)

                        keypoints = extract_keypoints(results)
                        npy_path = os.path.join(DATA_PATH, action, str(video), str(frame_num))
                        np.save(npy_path, keypoints)

                        if cv2.waitKey(10) & 0xFF == ord('q'):
                            break

        cap.release()
        cv2.destroyAllWindows()

    cap.release()
    cv2.destroyAllWindows()

def preprocess_data():
    global X_train, X_test, y_train, y_test
    label_map = {label:num for num, label in enumerate(actions)}

    sequences, labels = [], []

    for action in actions:
        for sequence in np.array(os.listdir(os.path.join(DATA_PATH, action))).astype(int):
            window = []
            for frame_num in range(frames_of_video):
                res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
                window.append(res)
            sequences.append(window)
            labels.append(label_map[action])

    X = np.array(sequences)
    y = to_categorical(labels).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)    
    
def build_network():
    log_dir = os.path.join('Logs')
    tb_callback = TensorBoard(log_dir=log_dir)

    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1662)))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(actions.shape[0], activation='softmax'))
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    model.fit(X_train, y_train, epochs=2000, callbacks=[tb_callback])
    model.save('action.h5')
    plot_model(model.summary(), to_file='model.png')

    
if __name__ == "__main__":
    global X_train, X_test, y_train, y_test
    training_testing()
    preprocess_data()
    build_network()
    
    model = load_model('action.h5')

    sequence = []
    sentence = []
    predictions = []
    threshold = 0.5

    cap = cv2.VideoCapture(0)
    
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()

            image, results = mediapipe_detection(frame, holistic)
                        
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-30:]
            
            if len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                predictions.append(np.argmax(res))
                
                
                if np.unique(predictions[-10:])[0]==np.argmax(res): 
                    if res[np.argmax(res)] > threshold: 
                        
                        if len(sentence) > 0: 
                            if actions[np.argmax(res)] != sentence[-1]:
                                sentence.append(actions[np.argmax(res)])
                        else:
                            sentence.append(actions[np.argmax(res)])

                if len(sentence) > 5: 
                    sentence = sentence[-5:]

                image = prob_viz(res, actions, image)
            
            cv2.imshow('OpenCV Feed', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()