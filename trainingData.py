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
            faceList = Image.open(f"{imgNPath}/{img}").convert("L")
            
            name = imgNPath.split("\\")
            
            faces.append(faceList)
            names.append(listname.index(name[1]))
            
    return faces,names

faces,names = toArrayImage(path)

names = np.array(names)
recognizer.train(faces,names)
recognizer.save("dataSet/training.yml")
cv2.destroyAllWindows()



