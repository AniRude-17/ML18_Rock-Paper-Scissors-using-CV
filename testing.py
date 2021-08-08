import cv2
import os
import sys
import numpy as np
import tensorflow
from tensorflow import keras

def predict_class(result):
    maxprob=max(result[0][0],result[0][1],result[0][2],result[0][3])
    if maxprob==result[0][3]:
        return("NOTHING",maxprob)
    elif maxprob==result[0][0]:
        return("Rock",maxprob)
    elif maxprob==result[0][1]:
        return("Paper",maxprob)
    elif maxprob==result[0][2]:
        return("Scissors",maxprob)

model=tensorflow.keras.models.load_model('final_modelv1.h5')


cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
start = False
count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        continue
    frame=cv2.flip(frame,1)
    cv2.rectangle(frame, (1000, 100), (700, 450), (255, 255, 255), 2)
    
    roi = frame[100:450,700:1000]
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    roi = cv2.resize(roi, (224, 224))
    result=model.predict(np.array([roi]))
    prediction,prob=predict_class(result)    
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame,prediction+" "+str(round(prob*100,2))+"%",
            (5, 50), font, 2.0, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.imshow("RPS TEST", frame)

    k = cv2.waitKey(10)

    if k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()