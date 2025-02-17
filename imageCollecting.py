import os
import cv2

DATA_DIR = './data'

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 36
dataset_size = 100
cap = cv2.VideoCapture(0)

for i in range(0, 36):
    class_dir = os.path.join(DATA_DIR, str(i))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print('Prikupljanje podataka za klasu {}'.format(i))

    done = False
    while True:
        ret, frame = cap.read()
        text = 'Skupljanje slika za klasu ' + format(i)
        cv2.putText(frame, 'Pritisnite "q" da biste zapoceli proces', (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                     0.8, (93, 26, 237), 1, cv2.LINE_AA)
        cv2.putText(frame, text, (20, 80), cv2.FONT_HERSHEY_SIMPLEX,
                     0.8, (93, 26, 237), 1, cv2.LINE_AA)
        cv2.imshow('Signify', frame)
        if cv2.waitKey(25) == ord('q'):
            break

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        cv2.imshow('Signify', frame)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(DATA_DIR, str(i), '{}.jpg'.format(counter)), frame)
        counter += 1 

cap.release()
cv2.destroyAllWindows()
