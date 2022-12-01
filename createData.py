import numpy as np
import cv2
import os
import random
import shutil
import face_recognition


path = "data"

classname = []

pathVd = []
pathFol = []
indexFol = []

for i in os.listdir(path):
    classname.append(i)
    pathSub = os.path.join(path,i)
    pathFol.append(pathSub)
    
    if os.path.exists ("train\\"+str(i)):
        shutil.rmtree(r'./train/'+str(i))
    os.mkdir("train\\"+str(i))

    for j in os.listdir(pathSub):
        pathJr = os.path.join(pathSub,j)

        pathVd.append(pathJr)
        count = 0
        cap = cv2.VideoCapture(pathJr)

        if not cap.isOpened():
            print("Cannot open camera")
            break
        success, rimage = cap.read()
        success = True
        while success:
            if count >= 150:
                break
            if not success:
                break
            pp = "train\\"+str(i)+"\\"+str(i)+"-"+str(count)+".jpg"
            cv2.imwrite(pp,rimage)
            success, rimage = cap.read()
            print("successfully!")
            count += 1
        count = 0