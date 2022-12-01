import numpy as np 
import pandas as pd 
import cv2
import time
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import re
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split , GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report , accuracy_score
import face_recognition

t1=time.time()
#folderpath="../input/pins-face-recognition/pins/PINS/"
#cascade = "../input/haarcascadefrontalfaces/haarcascade_frontalface_default.xml"
height=128
width=64
data=[]
labels=[]
Celebs=[]
for i in os.listdir("train"):

    path = os.path.join("train",i)
    for filename in os.listdir(path):
        img = np.array(cv2.imread(f"{path}/{filename}"))/255
        data.append(img)
        labels.append(i)
        
fig = plt.figure(figsize=(20,15))
print(len(labels))
for i in range(1,10):
    index = random.randint(0,20) #https://www.pythoncentral.io/how-to-generate-a-random-number-in-python/
    plt.subplot(3,3,i)
    plt.imshow(data[index])
    plt.xlabel(labels[index])
plt.show()
        