import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("./pk_1.jpg")

img = cv2.resize(img,None,fx=0.25,fy=0.25,interpolation=cv2.INTER_AREA)

img_g = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)


ret, threshold = cv2.threshold(img_g,130,255,0)

image, coutours, hierachy = cv2.findContours(threshold,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

for cnt in coutours:
    cv2.drawContours(img,[cnt],0,(0,0,255),1)

epsilon = 0.1 * cv2.arcLength(cnt,True)
approx = cv2.approxPolyDP(cnt,epsilon,True)




cv2.imshow("pk_1",img)
cv2.waitKey(0)