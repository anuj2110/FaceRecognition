# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 21:01:16 2020

@author: Arjun
"""

import cv2
from tensorflow.keras.models import load_model
import numpy as np
model_file_name = 'facefeatures_new_model.h5'
model = load_model(model_file_name)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

while True:
    ret,frame = cap.read()
    faces = face_cascade.detectMultiScale(frame,1.3,5)
    
    for (x,y,w,h) in faces:
        name = ''
        face = frame[y:y+h,x:x+w]
        face = cv2.resize(face,(224,224))
        face=np.array(face).reshape((1,224,224,3))/255.0
        predictions = model.predict(face)
        if(predictions[0][0]>0.5):
            name='Anuj'
        cv2.putText(frame,name, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()