import numpy as np
import cv2
from PIL import Image
import face_recognition
import os


labels = os.listdir("train")
#training hinh anh va thu vien nhan dien
recognizer = cv2.face.LBPHFaceRecognizer_create() # 
#face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
recognizer.read("dataSet/training2.yml")


# khoi dong webcam
cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read() # frame khung hinh, res bool
    if not ret:
        break
    frameS = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    # xac dinh vi tri khuan mat tai 1 frame bat ky
    faces = face_recognition.face_locations(frameS)
    # y1,x2,y2,x1 = facecurFrame[0]
    # y1,x2,y2,x1 = y1,x2,y2,x1
        
    # cv2.rectangle(frame,(x1,y1),(x2,y2),(128,64,255),2)
    # faces = face_cascade.detectMultiScale(frameS)
    
    for (y1,x2,y2,x1) in faces:
       
        cv2.rectangle(frame,(x1,y1),(x2, y2),(255,0,255),2)

        anh = frameS[y1:y2,x1:x2]
        
        ids,conf = recognizer.predict(anh)
        print(ids,conf)
        if conf < 60:
            name = labels[ids]
            cv2.putText(frame,name,(x1 + 10,y1  - 10),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,255),2)
        else:
            cv2.putText(frame,"unknown",(x1 + 10,y1  - 10),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
        
    cv2.imshow("face",frame)
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
        
        