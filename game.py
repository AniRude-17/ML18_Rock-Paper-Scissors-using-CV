import cv2
import os
import sys
import numpy as np
import tensorflow
from tensorflow import keras
import random
import winsound
from winsound import PlaySound, SND_FILENAME, SND_LOOP, SND_ASYNC


computer_choice_list=[0,1,2]
link_map={
    0:"Rock",
    1:"Paper",
    2:"Scissors",
    3:"NOTHING"
}
size=(300,350)

scissors_image=cv2.imread("media/Scissors.PNG")
scissors_image=cv2.resize(scissors_image,size,interpolation=cv2.INTER_LINEAR )
rock_image=cv2.imread("media/Rock.PNG")
rock_image=cv2.resize(rock_image,size,interpolation=cv2.INTER_LINEAR)
paper_image=cv2.imread("media/Paper.PNG")
paper_image=cv2.resize(paper_image,size,interpolation=cv2.INTER_LINEAR)
c_word=""
p_word=""

counter=0
temp=0
def computer_prediction():
    x=random.choice(computer_choice_list)
    return link_map[x]
    

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
rc=0
pc=0
sc=0
nc=0
rounds=0
p_score=0
c_score=0

while True:
    ret, frame = cap.read()
    if temp==0:
        winsound.PlaySound('media/audio.wav', winsound.SND_LOOP + winsound.SND_ASYNC)
        temp=1
    if not ret:
        continue
        
    if p_score==5 or c_score==5:
        cv2.rectangle(frame, (0, 0), (1280, 720), (0, 0, 0), -1)
        if c_score==5:
            cv2.putText(frame,"Computer Wins",(415, 400), font, 2.0, (255, 255, 255), 2, cv2.LINE_AA)
        else:
            cv2.putText(frame,"Player Wins",(450, 400), font, 2.0, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("RPS TEST", frame)
        
    else:
    
        frame=cv2.flip(frame,1)
        cv2.rectangle(frame, (100, 100), (400, 450), (0, 0, 0), 3)         #computer frame

        cv2.rectangle(frame, (1180, 100), (880, 450), (255, 255, 255), 3)  #player frame

        roi = frame[100:450,880:1180]
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        roi = cv2.resize(roi, (224, 224))
        if start:
            roi = frame[100:450,880:1180]
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            roi = cv2.resize(roi, (224, 224))
            count += 1
        result=model.predict(np.array([roi]))
        prediction,prob=predict_class(result)
        if (prediction!="NOTHING"):
            if (counter<=6):
                counter=rc+pc+sc
            if (not (counter>=7)):
                if (prediction=="Rock"):
                    rc=rc+1
                elif (prediction=="Paper"):
                    pc=pc+1
                else:
                    sc=sc+1
            if (counter==6):
                counter=counter+1
            if (counter==7):
                counter=8
                c_word=computer_prediction()
                if (rc>=3):
                    p_word="Rock"
                elif (pc>=3):
                    p_word="Paper"
                else:
                    p_word="Scissors"
                rounds=rounds+1

                cv2.putText(frame,c_word,
                    (190, 500), font, 2.0, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(frame,p_word,
                   (970, 500), font, 2.0, (255, 255, 255), 2, cv2.LINE_AA)
                if ((p_word=="Rock" and c_word=="Scissors") or (p_word=="Paper" and c_word=="Rock") or (p_word=="Scissors" and c_word=="Paper")):
                    cv2.rectangle(frame, (1190, 90), (870, 460), (255, 0, 0), 6)
                    p_score=p_score+1
                elif ((p_word=="Rock" and c_word=="Paper") or (p_word=="Paper" and c_word=="Scissors") or (p_word=="Scissors" and c_word=="Rock")):
                    cv2.rectangle(frame, (90, 90), (410, 460), (255, 0, 0), 6)
                    c_score=c_score+1
                else:
                    cv2.rectangle(frame, (1190, 90), (870, 460), (255, 0, 0), 6)
                    cv2.rectangle(frame, (90, 90), (410, 460), (255, 0, 0), 6)

                if (c_word=="Scissors"):
                    frame[100:450,100:400]=scissors_image
                if (c_word=="Rock"):
                    frame[100:450,100:400]=rock_image
                if (c_word=="Paper"):
                    frame[100:450,100:400]=paper_image

            if (counter==8):
                if ((p_word=="Rock" and c_word=="Scissors") or (p_word=="Paper" and c_word=="Rock") or (p_word=="Scissors" and c_word=="Paper")):
                    cv2.rectangle(frame, (1190, 90), (870, 460), (255, 0, 0), 6)
                elif ((p_word=="Rock" and c_word=="Paper") or (p_word=="Paper" and c_word=="Scissors") or (p_word=="Scissors" and c_word=="Rock")):
                    cv2.rectangle(frame, (90, 90), (410, 460), (255, 0, 0), 6)
                else:
                    cv2.rectangle(frame, (1190, 90), (870, 460), (255, 0, 0), 6)
                    cv2.rectangle(frame, (90, 90), (410, 460), (255, 0, 0), 6)

                cv2.putText(frame,c_word,
                    (190, 500), font, 2.0, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(frame,p_word,
                    (970, 500), font, 2.0, (255, 255, 255), 2, cv2.LINE_AA)

                if (c_word=="Scissors"):
                    frame[100:450,100:400]=scissors_image
                if (c_word=="Rock"):
                    frame[100:450,100:400]=rock_image
                if (c_word=="Paper"):
                    frame[100:450,100:400]=paper_image



        else:
            rc=0
            pc=0
            sc=0
            counter=0
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame," Computer : "+str(c_score)+"             Round : "+str(rounds)+"           Player :"+str(p_score),(100, 550), font, 1.10, (0,255,255), 2, cv2.LINE_AA)
        cv2.putText(frame,prediction,(1, 30), font, 1, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.imshow("RPS TEST", frame)

    k = cv2.waitKey(10)
    if k == ord('q'):
        break
        
winsound.PlaySound(None, SND_FILENAME)

cap.release()
cv2.destroyAllWindows()