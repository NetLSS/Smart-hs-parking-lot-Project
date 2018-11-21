import cv2

img = cv2.imread("./pk_1.jpg")  # ./ 현재 디렉토리에   pk_1.jpg 파일 불러오기

img = cv2.resize(img,None,fx=0.25,fy=0.25,interpolation=cv2.INTER_AREA)  # 크기가 너무 커서 1/4 로 resize

img_g = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)  # rgb to gray
#영역자르기
img_c = img_g[220:1000,0:900]  # cropping ( 주차장 영역만 잘라서 영상처리 )

#binary이미지로 변환
ret, threshold = cv2.threshold(img_c,200,255,0) # gray값 200~255 를 1로  나머지를 0으로 binary img 생성  
                                                # 예제 참고하여 ret 은 있으나 사용안함
# 선 탐색
image, coutours, hierachy = cv2.findContours(threshold,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)  # threshold 이미지 (binary 이미지) 를
                                                    # cv2.RETR_TREE : contours line을 찾으며, 모든 hieracy관계를 구성함
                                                    # cv2.CHAIN_APPROX_SIMPLE :  contours line을 그릴 수 있는 point 만 저장. (ex; 사각형이면 4개 point)

#영역을 자를 때 220~1000 (y좌표) 를 사용했으므로   y좌표에 +220을 해준다.
for cnt in coutours:
    for i in range(0, len(cnt)):
         print(i,end=' ')
         cnt[i][0][1] = cnt[i][0][1] + 220
         print(cnt[i][0][1])

# 빨간 선 그리기           
for cnt in coutours:
    cv2.drawContours(img,[cnt],0,(0,0,255),1)
#findContours, drawCoutours 참조: https://opencv-python.readthedocs.io/en/latest/doc/15.imageContours/imageContours.html 참조


epsilon = 0.1 * cv2.arcLength(cnt,True)
approx = cv2.approxPolyDP(cnt,epsilon,True)




#cv2.imshow("crop",img_c)
cv2.imshow("pk_1",img)
cv2.waitKey(0)
