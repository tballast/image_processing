# simple webcam face detection
# shows live webcam with square around face, uses all haarcascades available
import numpy as np
import cv2
face_cascade = {}
eye_cascade = {}

face_cascade[1] = cv2.CascadeClassifier('/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')
face_cascade[2] = cv2.CascadeClassifier('/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml')
face_cascade[3] = cv2.CascadeClassifier('/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_alt2.xml')
face_cascade[4] = cv2.CascadeClassifier('/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_alt_tree.xml')
face_cascade[5] = cv2.CascadeClassifier('/usr/local/share/OpenCV/haarcascades/haarcascade_profileface.xml')
eye_cascade = cv2.CascadeClassifier('/usr/local/share/OpenCV/haarcascades/haarcascade_eye.xml')

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
	ret, frame = cap.read()

    # Our operations on the frame come here
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	for k in range(1,len(face_cascade)):
		faces = face_cascade[k].detectMultiScale(gray, 1.3, 5)
	
		for (x,y,w,h) in faces:
			cv2.rectangle(gray,(x,y),(x+w,y+h),(255,0,0),2)
			roi_gray = gray[y:y+h, x:x+w]
			roi_color = frame[y:y+h, x:x+w]
		# eyes = eye_cascade.detectMultiScale(roi_gray)
		# for (ex,ey,ew,eh) in eyes:
		# 	cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

	# cv2.namedWindow("img",cv2.WINDOW_NORMAL)
	# cv2.imshow('img',img)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()



    # Display the resulting frame
	cv2.imshow('frame',gray)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
