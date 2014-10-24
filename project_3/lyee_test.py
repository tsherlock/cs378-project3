#!/lusr/bin/python

import numpy as np
import cv2

cap = cv2.VideoCapture('test_data/ball_2.mov')

fgbg = cv2.BackgroundSubtractorMOG()

while(1):
    ret, frame = cap.read()

    fgmask = fgbg.apply(frame)
    
    im = fgmask
    #print fgmask.shape
    
    im2 = fgmask #cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(im2,127,255,0)
    contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    for i in range(0, len(contours)):
        if (i % 2 == 0):
           cnt = contours[i]
           x,y,w,h = cv2.boundingRect(cnt)
           print x,y,w,h
           cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),2)
        
    cv2.imshow('frame',im2)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
