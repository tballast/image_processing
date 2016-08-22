# motion detecting. 
# watches webcam, detects motion at 2fps, then increases to 30 to capture video

import numpy as np
import cv2
import time


# start video capture
cap = cv2.VideoCapture(0)
# create forground background difference
fgbg = cv2.createBackgroundSubtractorMOG2()
# define output
fourcc = cv2.VideoWriter_fourcc(*'XVID')


# threshold for motion (% of frame)
thresh_motion = 20
# threshold for time to save image (seconds)
thresh_time = 5


# continually
while(True):
	# delay by 400 ms
	time.sleep(0.1)

	# get frame
	ret, frame = cap.read()

	# apply foregroundbackground to frame
	fgmask = fgbg.apply(frame)

	# if motion detected (10% of frame)
	if np.mean(fgmask) > thresh_motion:
		print 'starting capture'

		# specify out image with time activated
		out = cv2.VideoWriter(str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())) + '.avi',fourcc,20.0,(640,480))
		# get time
		t1 = time.time()
		cat = 0

		while(cap.isOpened()):
			# get new frame
			ret, frame = cap.read()

			fgmask = fgbg.apply(frame)

			if ret == True:
				cv2.putText(frame, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),(10,frame.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX,0.35,(0,0,0),1)
				out.write(frame)

			# show frame mask that was calculated, quits when you press q
			cv2.imshow('frame',fgmask)

			# if there is no longer motion 
			if (np.mean(fgmask) < thresh_motion):
				cat += 1
				# if this notice has been triggered
				if cat > 0:
					# if the time between now and t1 is greather than a threshold
					if ((time.time() - t1) > thresh_time):
						# end
						print 'ending capture'
						break
			# if there is motion or you don't meet the other requirements
			else:
				cat = 0
				t1 = time.time()

			# if (np.mean(fgmask) < thresh_motion) and ((time.time() - t1) > thresh_time):
			# 	print 'ending capture'
			# 	break


			# show frame mask that was calculated, quits when you press q
			cv2.imshow('frame',fgmask)
			k = cv2.waitKey(1) & 0xff
			if k == ord('q'):
				break


# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

