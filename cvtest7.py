# teach kristen's face, find in webcam
# based off cvtest5

import numpy as np 
import cv2
import os
from PIL import Image

# get the path to a general face recognition
cascadePath =  '/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml'
# get the face cascade from the general
faceCascade = cv2.CascadeClassifier(cascadePath)

# define the recognizer to use
recognizer = cv2.face.createLBPHFaceRecognizer()



# define function to prepare training sets
def get_images_and_labels(path):
	# append all the absolute image paths in a list image_paths
	# we will not read the image with the .sad extension in the training set
	# rather, we will use them to test the accuracy of the training
	image_paths = [os.path.join(path,f) for f in os.listdir(path)]
	# images will contain face images
	images = []
	# labels will contain the label that is assigned to the image
	labels = []
	# loop through each image
	for image_path in image_paths:
		# read the image and convert to greyscale
		# image_pil = Image.open(image_path).convert('L')
		# convert the image format into numpy 
		# image = np.array(image_pil, 'uint8')
		img = cv2.imread(image_path)
		image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		# get the label of the image
		nbr = 1
		# detect the face in the image
		faces = faceCascade.detectMultiScale(image,1.5,5)
		# if face is detected, append the face to images and the label to labels
		for (x,y,w,h) in faces:
			images.append(image[y:y+h, x:x+w])
			# images.append( cv2.resize(image[y:y+h, x:x+w],(640,360)) )
			labels.append(nbr)
			cv2.namedWindow("Adding faces to training set...",cv2.WINDOW_NORMAL)
			cv2.imshow("Adding faces to training set...", image[y:y+h, x:x+w])
			cv2.waitKey(5)
	# return the images list and labels list
	return images, labels







## teach the face reconizer

# get path to faces
path = '/home/kris/Pictures/krisfaces'

# call the get_images_and_labels function
images,labels = get_images_and_labels(path)
cv2.destroyAllWindows()

# perform the training
recognizer.train(images,np.array(labels))


# start capturing
cap = cv2.VideoCapture(0)



	# predict_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	# # find the face in the image to get its ROI
	# faces = faceCascade.detectMultiScale(predict_image,1.3,5)
	# for (x,y,w,h) in faces:
	# 	# see if you can regognize the face
	# 	nbr_predicted,conf = recognizer.predict(predict_image[y:y+h, x:x+w])
	# 	# nbr_predicted,conf = recognizer.predict( cv2.resize(predict_image[y:y+h, x:x+w],(640,360)) )
	# 	cv2.rectangle(predict_image,(x,y),(x+w,y+h),(255,0,0),2)
	# 	# roi_gray = gray[y:y+h, x:x+w]
	# 	# roi_color = frame[y:y+h, x:x+w]
	# 	print image_path
	# 	print nbr_predicted, conf
	# 	# nbr_actual = os.path.split(image_path)[2]
	# 	# if nbr_actual == nbr_predicted:
	# 		# print "{} is correctly recognized with confidence {}".format(nbr_actual,conf)
	# 	# else:
	# 		# print "{} is incorrectly recognized as {}".format(nbr_actual,nbr_predicted)
	# 	cv2.namedWindow("Image",cv2.WINDOW_NORMAL)
	# 	cv2.imshow("Image",predict_image)
	# 	cv2.waitKey(1000)




while(True):
    # Capture frame-by-frame
	ret, frame = cap.read()

    # Our operations on the frame come here
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


	faces = faceCascade.detectMultiScale(gray, 1.3, 5)
	
	for (x,y,w,h) in faces:
		nbr_predicted,conf = recognizer.predict(gray[y:y+h, x:x+w])
		cv2.rectangle(gray,(x,y),(x+w,y+h),(255,0,0),2)
			# roi_gray = gray[y:y+h, x:x+w]
			# roi_color = frame[y:y+h, x:x+w]
		# eyes = eye_cascade.detectMultiScale(roi_gray)
		# for (ex,ey,ew,eh) in eyes:
		# 	cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

	# cv2.namedWindow("img",cv2.WINDOW_NORMAL)
	# cv2.imshow('img',img)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()



    # Display the resulting frame
	cv2.imshow('frame',gray)
	if len(faces) > 0:
		print nbr_predicted, conf
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
