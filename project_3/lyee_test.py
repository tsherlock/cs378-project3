#!/lusr/bin/python

import numpy as np
import cv2

cap = cv2.VideoCapture('test_data/ball_3.mov')

fgbg = cv2.BackgroundSubtractorMOG(200, 10, 0.7, 0)

ret, frame = cap.read()
init_fgmask = fgbg.apply(frame)
while(1):
    ret, frame = cap.read()
    fgmask = fgbg.apply(frame)
    
    im = fgmask - init_fgmask
    
    # im2 = cv2.filter2D(frame, -1, fgmask) #cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # ret,thresh = cv2.threshold(im,127,255,cv2.THRESH_BINARY)
    # contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    # print "contours ", len(contours)
    # for i in range(0, len(contours)):
    #   # if (i % 2 == 0):
    #   cnt = contours[i]
    #   x,y,w,h = cv2.boundingRect(cnt)
    #   # print x,y,w,h
    #   cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),2)
    cv2.imshow('frame',im)
    # cv2.imshow('original', frame)
    k = cv2.waitKey(500) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
