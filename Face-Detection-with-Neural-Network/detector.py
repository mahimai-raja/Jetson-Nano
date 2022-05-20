import cv2 as cv
import numpy as np
from tensorflow.keras.models import model_from_json

prediction_dict = {0:'kalam',
                1: 'mahimai'
                            }


json_file = open('model_structure.json','r')
model_structure = json_file.read()
json_file.close()

model = model_from_json(model_structure)

model.load_weights('model_weights.h5')

video = cv.VideoCapture(0)

while True :
    return_value, frame = video.read()
    frame = cv.resize(frame, (1280, 720))
    
    face_detector = cv.CascadeClassifier('cascade.xml')
    grey = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    detection = face_detector.detectMultiScale(grey,
                                    scaleFactor=1.3,
                                    minNeighbors=5)

    for (x, y, w, h) in detection:
        cv.rectangle(frame, 
                (x, y-50),
                (x+w, y+h+10),
                (0,255,0),4)
        
        faces = grey[ y: y+h, x: x+w ]
        resized_face = np.expand_dims(np.expand_dims(cv.resize(faces,(48,48)),-1),0)
        print(resized_face)
        prediction = model.predict(resized_face)
        maxindex = np.argmax(prediction)
        print(prediction_dict[maxindex])
        
        cv.putText(frame,
                prediction_dict[maxindex], 
                (x, y-10),
                cv.FONT_HERSHEY_SIMPLEX, 1,
                (150,150,),2,
                cv.LINE_AA)

    cv.imshow('Face Detection', frame)
    
    if cv.waitKey(1) % 256 == 27 :
        break

video.release()
cv.destroyAllWindows()