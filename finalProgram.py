import pickle
import numpy as np
import cv2
import mediapipe as mp

with open('./model.p', 'rb') as f:  
    model_dict = pickle.load(f)
model = model_dict['model']

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F',
    6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L',
    12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R',
    18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X',
    24: 'Y', 25: 'Z', 26: '0', 27: '1', 28: '2', 29: '3',
    30: '4', 31: '5', 32: '6', 33: '7', 34: '8', 35: '9'
}

while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()

    if not ret:
        break

    H, W, _ = frame.shape
    center_x, center_y = W / 2, H / 2  

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        closest_hand_index = None
        closest_distance = float('inf')

        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            landmarks = hand_landmarks.landmark
            cx = np.mean([landmarks[j].x for j in range(len(landmarks))])
            cy = np.mean([landmarks[j].y for j in range(len(landmarks))])

            distance = np.sqrt((cx - center_x) ** 2 + (cy - center_y) ** 2)

            if distance < closest_distance:
                closest_distance = distance
                closest_hand_index = i

        hand_landmarks = results.multi_hand_landmarks[closest_hand_index]

        mp_drawing.draw_landmarks(
            frame, 
            hand_landmarks,  
            mp_hands.HAND_CONNECTIONS, 
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style()
        )

        for landmark in hand_landmarks.landmark:
            x = landmark.x
            y = landmark.y

            x_.append(x)
            y_.append(y)

        for landmark in hand_landmarks.landmark:
            x = landmark.x
            y = landmark.y
            data_aux.append(x - min(x_))
            data_aux.append(y - min(y_))

        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10
        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

        prediction = model.predict([np.asarray(data_aux)])
        predicted_character = labels_dict[int(prediction[0])]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, predicted_character, (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (93, 26, 237), 1, cv2.LINE_AA)

    cv2.imshow('Signify', frame)

    if cv2.waitKey(1) & 0xFF == 27:  
        break

cap.release()
cv2.destroyAllWindows()
