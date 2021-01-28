import cv2
import face_recognition
import numpy as np
import os


#convert the images into RGB bcoz the package takes RGB while the load inputs BGR


#load the images
elontrain=face_recognition.load_image_file('imageset/elon.jpg')
elontrain=cv2.cvtColor(elontrain,cv2.COLOR_BGR2RGB)

elontest=face_recognition.load_image_file('imageset/elon2.jpg')
elontest=cv2.cvtColor(elontest,cv2.COLOR_BGR2RGB)

faceloc=face_recognition.face_locations(elontrain)[0]
train_encod=face_recognition.face_encodings(elontrain)[0]
cv2.rectangle(elontrain,(faceloc[3],faceloc[0]),(faceloc[1],faceloc[2]),(0,255,255),4)


faceloctest=face_recognition.face_locations(elontest)[0]
test_encod=face_recognition.face_encodings(elontest)[0]
cv2.rectangle(elontest,(faceloctest[3],faceloctest[0]),(faceloctest[1],faceloctest[2]),(0,255,255),4)


result=face_recognition.compare_faces([train_encod],test_encod)
dis=face_recognition.face_distance([train_encod],test_encod)
print(result,dis)
cv2.putText(elontest,f'{result} {dis}',(50,50),cv2.FONT_HERSHEY_DUPLEX,1,(0,0,255),3)


cv2.imshow('elon train',elontrain)
cv2.waitKey(0)

cv2.imshow('elon test',elontest)
cv2.waitKey(0)
