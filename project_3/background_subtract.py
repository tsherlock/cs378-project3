#!/lusr/bin/python

import numpy as np
import cv2

imnames = []

path = "test_data/ball_4_frames/"
i = 0
while i <= 326:
	s = str(i)
	while len(s) < 3:
		s = "0" + s
	imnames.append(path + "frame_" + s + ".png")
	i = i + 1

image1 = cv2.imread(imnames[200])
image2 = cv2.imread(imnames[201])

rows,cols,_ = image2.shape

offset = -1

M = np.float32([[1,0,offset],
                [0,1,0]])
image2 = cv2.warpAffine(image2,M,(cols,rows))

image3 = image1 - image2

image3 = cv2.cvtColor(image3,cv2.COLOR_BGR2GRAY)
image3 = cv2.adaptiveThreshold(image3,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
contours, _ = cv2.findContours(image3, cv2.RETR_TREE,
                                       cv2.CHAIN_APPROX_SIMPLE)

sort_cnt = sorted(contours, key=lambda x: cv2.boundingRect(x)[2],
                  reverse=True)

x, y, w, h = cv2.boundingRect(sort_cnt[0])
# result.append((x, y, x+w, y+h))
cv2.rectangle(image3, (x, y), (x+w, y+h), (255, 255, 255), 2)

cv2.imshow('subtracted image', image3)
cv2.imwrite('subtracted_img' + str(offset) + ".png", image3)
cv2.waitKey()
