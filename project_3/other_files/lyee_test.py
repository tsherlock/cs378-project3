#!/lusr/bin/python

import numpy as np
import cv2

def findBackground(video):
    cap = cv2.VideoCapture(video)

    # fgbg = cv2.BackgroundSubtractorMOG(200, 10, 0.7, 0)
    _ , result = cap.read()
    result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    # init_fgmask = fgbg.apply(frame)
    i = 1
    n = 100
    while(i < n):
        i += 1
        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        beta = 1.0/i
        alpha = 1.0 - beta
        # print alpha, beta
        result = cv2.addWeighted(result, alpha, frame, beta, 0.0)

    _ , result = cv2.threshold(result,127,255,cv2.THRESH_BINARY)
    # cv2.imshow('average background',result)
    # k = cv2.waitKey(2000) & 0xff

    cap.release()
    cv2.destroyAllWindows()
    return result

video = 'test_data/ball_3.mov'
cap = cv2.VideoCapture(video)

result = []

background = findBackground(video)
fgbg = cv2.BackgroundSubtractorMOG(200, 10, 0.7, 0)
fgbg_init = fgbg.apply(background)
while(1):
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = fgbg.apply(frame)

    ret,thresh = cv2.threshold(frame,220,255,cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    # print "contours ", len(contours)
    for i in range(0, len(contours)):
      cnt = contours[i]
      x,y,w,h = cv2.boundingRect(cnt)
      # print x, y, w ,h
      if w > 30:
        result.append((x,y,x+w,y+h))
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
    
    cv2.imshow('frame',frame)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()




