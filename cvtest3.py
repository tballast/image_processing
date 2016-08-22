# motion detecting. 
# watches webcam, detects motion

import numpy as np
import cv2
import time


# start video capture
cap = cv2.VideoCapture(0)
# create forground background difference
fgbg = cv2.createBackgroundSubtractorMOG2()



# continually
while(True):
	# get frame
	ret, frame = cap.read()

	# apply foregroundbackground to frame
	fgmask = fgbg.apply(frame)

	# print average of frame
	# print np.mean(fgmask)



	# show frame mask that was calculated, quits when you press q
	cv2.imshow('frame',fgmask)
	k = cv2.waitKey(1) & 0xff
	if k == ord('q'):
		break


# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

