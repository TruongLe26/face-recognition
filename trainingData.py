import os
import numpy as np
import cv2
from PIL import Image
import face_recognition


recognizer = cv2.face.LBPHFaceRecognizer_create() # 
path = "train"

def toArrayImage(path):
    namePath = os.listdir(path)
    faces = []
    names = []
    listname = []
    for nimg in namePath:
        imgNPath = os.path.join(path,nimg)
        listname.append(imgNPath.split("\\")[1])
        for img in os.listdir(imgNPath):
            face = Image.open(f"{imgNPath}/{img}").convert("L")
            faceList = np.array(face)
            name = imgNPath.split("\\")
            
            faces.append(faceList)
            names.append(listname.index(name[1]))
            
    return faces,names

faces,names = toArrayImage(path)

names = np.array(names)

def encodeFace(faces):
    encode = []
    for i in faces:
        face = cv2.cvtColor(i,cv2.COLOR_BGR2RGB)
        enImg = face_recognition.face_encodings(face)
        encode.append(enImg)
    return encode

encodeFaces = encodeFace(faces)
print(type(encodeFaces))
# recognizer.train(faces,names)
# recognizer.save("dataSet/training.yml")
np.save("text1.txt",encodeFaces)
cv2.destroyAllWindows()



