# experimental face recognition from http://hanzratech.in/2015/02/03/face-recognition-using-opencv.html

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
	image_paths = [os.path.join(path,f) for f in os.listdir(path) if not f.endswith('.sad')]

	# images will contain face images
	images = []
	# labels will contain the label that is assigned to the image
	labels = []

	# loop through each image
	for image_path in image_paths:
		# read the image and convert to greyscale
		image_pil = Image.open(image_path).convert('L')
		# convert the image format into numpy 
		image = np.array(image_pil, 'uint8')
		# get the label of the image
		nbr = int(os.path.split(image_path)[1].split(".")[0].replace("subject",""))
		# detect the face in the image
		faces = faceCascade.detectMultiScale(image)
		# if face is detected, append the face to images and the label to labels
		for (x,y,w,h) in faces:
			images.append(image[y:y+h, x:x+w])
			labels.append(nbr)
			cv2.imshow("Adding faces to training set...", image[y:y+h, x:x+w])
			cv2.waitKey(50)

	# return the images list and labels list
	return images, labels







## teach the face reconizer

# get path to yale dataset
path = '/home/kris/Pictures/yalefaces/yalefaces'

# call the get_images_and_labels function
images,labels = get_images_and_labels(path)
cv2.destroyAllWindows()

# perform the training
recognizer.train(images,np.array(labels))




## test the face recognizer

# append the images with the .sad extension into image_paths
image_paths = [os.path.join(path,f) for f in os.listdir(path) if f.endswith('sad')]
# go through each image
for image_path in image_paths:
	# open the image and conert
	predict_image_pil = Image.open(image_path).convert('L')
	predict_image = np.array(predict_image_pil,'uint8')
	# find the face in the image to get its ROI
	faces = faceCascade.detectMultiScale(predict_image)
	for (x,y,w,h) in faces:
		# see if you can regognize the face
		nbr_predicted,conf = recognizer.predict(predict_image[y:y+h, x:x+w])
		nbr_actual = int(os.path.split(image_path)[1].split(".")[0].replace("subject",""))
		if nbr_actual == nbr_predicted:
			print "{} is correctly recognized with confidence {}".format(nbr_actual,conf)
		else:
			print "{} is incorrectly recognized as {}".format(nbr_actual,nbr_predicted)
		cv2.imshow("Recognizing Face",predict_image[y:y+h,x:x+w])
		cv2.waitKey(1000)

