import numpy as np
import cv2

face_cascade = {}
eye_cascade = {}

# simple face detector, uses input image

face_cascade[1] = cv2.CascadeClassifier('/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')
face_cascade[2] = cv2.CascadeClassifier('/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml')
face_cascade[3] = cv2.CascadeClassifier('/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_alt2.xml')
face_cascade[4] = cv2.CascadeClassifier('/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_alt_tree.xml')
face_cascade[5] = cv2.CascadeClassifier('/usr/local/share/OpenCV/haarcascades/haarcascade_profileface.xml')
eye_cascade[1] = cv2.CascadeClassifier('/usr/local/share/OpenCV/haarcascades/haarcascade_eye.xml')
eye_cascade[2] = cv2.CascadeClassifier('/usr/local/share/OpenCV/haarcascades/haarcascade_eye_tree_eyeglasses.xml')
eye_cascade[3] = cv2.CascadeClassifier('/usr/local/share/OpenCV/haarcascades/haarcascade_lefteye_2splits.xml')
eye_cascade[4] = cv2.CascadeClassifier('/usr/local/share/OpenCV/haarcascades/haarcascade_righteye_2splits.xml')

img = cv2.imread('rustest.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)



for k in range(1,len(face_cascade)):
	faces = face_cascade[k].detectMultiScale(gray, 1.3, 5)



	for (x,y,w,h) in faces:
		cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
		roi_gray = gray[y:y+h, x:x+w]
		roi_color = img[y:y+h, x:x+w]
    # eyes = eye_cascade.detectMultiScale(roi_gray)
    # for (ex,ey,ew,eh) in eyes:
    #     cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

cv2.namedWindow("img",cv2.WINDOW_NORMAL)
cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
# print faces