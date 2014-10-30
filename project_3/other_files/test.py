#!/lusr/bin/python

import numpy as np
import cv2

cap = cv2.VideoCapture('test_data/ball_3.mov')

# fgbg = cv2.BackgroundSubtractorMOG(200, 10, 0.7, 0)
_ , result = cap.read()
result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
# init_fgmask = fgbg.apply(frame)
i = 1
n = 30
while(i < n):
	i += 1
	ret, frame = cap.read()
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	beta = 1.0/i
	alpha = 1.0 - beta
	print alpha, beta
	result = cv2.addWeighted(result, alpha, frame, beta, 0.0)

_ , result = cv2.threshold(result,220,255,cv2.THRESH_BINARY)
cv2.imshow('average background',result)
k = cv2.waitKey(2000) & 0xff

cap.release()
cv2.destroyAllWindows()
