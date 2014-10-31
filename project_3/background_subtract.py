#!/lusr/bin/python

import numpy as np
import cv2

image1 = cv2.imread("188.png")
image2 = cv2.imread("187.png")

rows,cols,_ = image2.shape

offset = 1

M = np.float32([[1,0,offset],
                [0,1,0]])
image2 = cv2.warpAffine(image2,M,(cols,rows))

image3 = image1 - image2

cv2.imshow('subtracted image', image3)
cv2.imwrite('subtracted_img' + str(offset) + ".png", image3)
cv2.waitKey()
