import cv2

img = cv2.imread("./pk_1.jpg")

img = cv2.resize(img,None,fx=0.25,fy=0.25,interpolation=cv2.INTER_AREA)

img_g = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
img_c = img_g[220:1000,0:900]
ret, threshold = cv2.threshold(img_c,200,255,0)

image, coutours, hierachy = cv2.findContours(threshold,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

for cnt in coutours:
    for i in range(0, len(cnt)):
         print(i,end=' ')
         cnt[i][0][1] = cnt[i][0][1] + 220
         print(cnt[i][0][1])
for cnt in coutours:
    cv2.drawContours(img,[cnt],0,(0,0,255),1)


epsilon = 0.1 * cv2.arcLength(cnt,True)
approx = cv2.approxPolyDP(cnt,epsilon,True)




#cv2.imshow("crop",img_c)
cv2.imshow("pk_1",img)
cv2.waitKey(0)