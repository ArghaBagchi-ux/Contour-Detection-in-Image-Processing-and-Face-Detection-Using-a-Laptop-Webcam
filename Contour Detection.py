#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import matplotlib.pyplot as plt


# In[5]:


Gray_image = cv2.imread("C:/Users/User/Downloads/0_0_0.png",0)
plt.figure(figsize=[10,10])
plt.imshow(Gray_image,cmap='gray');plt.title('Original');plt.axis('off');


# In[7]:


#Find all conturs in the image

contours,hierarchy = cv2.findContours(Gray_image,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
print(format(len(contours)))


# In[8]:


image_copy = cv2.imread("C:/Users/User/Downloads/0_0_0.png")
cv2.drawContours(image_copy,contours,-1,(0,255,0),3)
plt.figure(figsize=[10,10])
plt.imshow(image_copy[:,:,::-1]);plt.axis('off');


# In[9]:


image2=cv2.imread('"C:/Users/User/Downloads/0_0_0.png"')
plt.imshow(image2[:,:,::-1]);plt.title('Original_image');plt.axis('off');


# In[10]:


import cv2
import matplotlib.pyplot as plt

# Correct the file path
image2 = cv2.imread("C:/Users/User/Downloads/0_0_0.png")

# Check if the image was read correctly
if image2 is not None:
    # Convert the image from BGR (OpenCV format) to RGB (Matplotlib format)
    plt.imshow(image2[:,:,::-1])
    plt.title('Original Image')
    plt.axis('off')
    plt.show()
else:
    print("Error: Image not found or unable to read.")


# In[1]:


import cv2
import numpy

#Read image
image = cv2.imread("C:/Users/User/Downloads/0_0_0.png")
image = cv2.resize(image,None,fx=0.9,fy=0.9)

gray = cv2.cvtColor(image,cv2. COLOR_BGR2GRAY)

#NOW GRAYSCALE IMG TO BYNARY IMG
ret,binary=cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

#NOW DETECT THE CONTOUR
contours,hierarchy = cv2.findContours(binary,mode = cv2.RETR_TREE,method=cv2.CHAIN_APPROX_NONE)

#VISUALIZE THE DATA STRUCTURE
print('Length of contours {}'.format(len(contours)))
print(contours)

#draw contours on the original image
image_copy = image.copy()
image_copy = cv2.drawContours(image_copy,contours,-1,(0,255,0),thickness=2,lineType=cv2.LINE_AA)


#RESULT
cv2.imshow('Grayscale Image',gray)
cv2.imshow('Drawn Contours',image_copy)
cv2.imshow('Binary Image',binary)

cv2.waitKey(0)
cv2.destroyAllWindows()


# In[18]:


import cv2
import numpy

#Read image
image = cv2.imread('C:/Users/User/Downloads/a-Original-MRI-brain-tumor-image-b-Colored-MRI-image.png')
image = cv2.resize(image,None,fx=0.9,fy=0.9)

gray = cv2.cvtColor(image,cv2. COLOR_BGR2GRAY)

#NOW GRAYSCALE IMG TO BYNARY IMG
ret,binary=cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

#NOW DETECT THE CONTOUR
#contours,hierarchy = cv2.findContours(binary,mode = cv2.RETR_TREE,method=cv2.CHAIN_APPROX_NONE)
#contours,hierarchy = cv2.findContours(binary,mode = cv2.RETR_LIST,method=cv2.CHAIN_APPROX_NONE)
contours,hierarchy = cv2.findContours(binary,mode = cv2.RETR_EXTERNAL,method=cv2.CHAIN_APPROX_NONE)

#VISUALIZE THE DATA STRUCTURE
print('Length of contours {}'.format(len(contours)))
print(contours)

#draw contours on the original image
image_copy = image.copy()
image_copy = cv2.drawContours(image_copy,contours,-1,(0,255,0),thickness=2,lineType=cv2.LINE_AA)


#RESULT
cv2.imshow('Grayscale Image',gray)
cv2.imshow('Drawn Contours',image_copy)
cv2.imshow('Binary Image',binary)

cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:


##Face and Eye Detection in image


# In[8]:


#Face Detection using haarcascade file 
import cv2
import numpy
face=cv2.CascadeClassifier("C:/Users/User/Downloads/archive (7)/haarcascade_frontalface_alt.xml") #for detecting face
eye = cv2.CascadeClassifier("C:/Users/User/Downloads/archive (7)/haarcascade_eye.xml") #for detecting eyes

image=cv2.imread("C:/Users/User/Downloads/e2f24de0-810e-400d-b9a5-79f3b83e41c1.jpg")
gray= cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) #convert into gray 

#parameters(img,scale_factor[reduce image size],min_neighbour)
faces = face.detectMultiScale(gray,4,4)   #for  faces

for(x,y,w,h) in faces:
    
    image=cv2.rectangle(image,(x,y),(x+w,y+h),(127,0,205),3)
    
    #Now detect eyes
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = image[y:y+h, x:x+w]
    eyes = eye.detectMultiScale(roi_gray,1.2,1)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(255,0,0),2)
    
image = cv2.resize(image,(800,700))
cv2.imshow("Face Detected",image)
cv2.waitKey(0)
cv2.destroyAllWindows()  


# In[ ]:


# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 18:43:51 2020

@author: NISHANT
"""
import cv2
import numpy
face=cv2.CascadeClassifier("C:/Users/User/Downloads/archive (7)/haarcascade_frontalface_alt.xml")
eye = cv2.CascadeClassifier("C:/Users/User/Downloads/archive (7)/haarcascade_eye.xml") #for detecting eyes
def dector(img):
    gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face.detectMultiScale(gray,1.3,5)
    
    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(127,0,125),3)
        
        roi_gray = gray[y:y+h, x:x+w]
        
        roi_color = img[y:y+h, x:x+w]
        
        eyes = eye.detectMultiScale(roi_gray,1.3,3)
        for (ex,ey,ew,eh) in eyes:
            cv2.circle(roi_color,(ex+27,ey+27),20,(255,255,0),2)

    return img

cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
while True:
    ret,frame =cap.read()
    frame = cv2.flip(frame,2)
    cv2.imshow("face dect",dector(frame))
    if cv2.waitKey(1)==13:   # press enter to terminate
        break
    
cap.release()
cv2.destroyAllWindows()  


# In[ ]:




