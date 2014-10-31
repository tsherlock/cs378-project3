#!/lusr/bin/python

import numpy as np
import cv2

def readImg(imgName, num):
  im = cv2.imread(imgName)
  
  im2 = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
  ret,thresh = cv2.threshold(im2,127,255,0)
  contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
  
  for i in range(0, len(contours)):
    cnt = contours[i]
    x,y,w,h = cv2.boundingRect(cnt)
    cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),2)
  cv2.imwrite(str(num)+'.png', im)


imnames = []
path = "test_data/ball_4_frames/"
i = 0
while i <= 326:
  s = str(i)
  while len(s) < 3:
    s = "0" + s
  imnames.append(path + "frame_" + s + ".png")
  i = i + 1
num = 0
for name in imnames:
  readImg(name, num)
  num = num + 1
