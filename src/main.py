# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 13:21:17 2015

@author: elad
"""

"""
Edit on Tue Nov 15 20:22:02 2018

@author : LeeSangSu
Log
181115 : 코드 분석 및 한국어 주석작업.
"""
import yaml
import numpy as np
import cv2

fn = r"C:\Users\elad\Documents\code\illumasense\datasets\CUHKSquare.mpg"           # 사용될 영상
# fn = r"C:\Users\elad\Documents\code\illumasense\datasets\parkinglot_1_720p.mp4"
# fn = r"C:\Users\elad\Documents\code\illumasense\datasets\street_high_360p.mp4"
fn_yaml = r"C:\Users\elad\Documents\code\illumasense\datasets\CUHKSquare.yml"      # 사용될 yml (yml 이란 :  '사람이 쉽게 읽을 수 있는' 데이터 직렬화 양식)
"""
이때 CUHKSquare.yml 는 아래와 같다.
-
    id: 0
    points: [[47,278],[63,246],[40,249],[12,280]]
-
    id: 1
    points: [[109,281],[73,283],[99,250],[127,250]]
-
    id: 2
    points: [[184,254],[164,284],[134,288],[152,256]]
-
    id: 3
    points: [[219,251],[255,252],[230,292],[192,292]]
-
    id: 4
    points: [[275,246],[325,249],[305,293],[261,293]]
"""
fn_out = r"C:\Users\elad\Documents\code\illumasense\datasets\output4.avi" # 결과물로 반환될 영상(?)
config = {'save_video': False,            # 비디오 저장
          'text_overlay': True,           # 텍스트 오버레이
          'parking_overlay': True,        # 파킹 오버레이
          'parking_id_overlay': True,     # 파킹 아이디 오버레이
          'parking_detection': True,      # 파킹 감지
          'motion_detection': False,      # 동작 감지
          'pedestrian_detction': False,   # 보행자 감지
          'min_area_motion_contour': 150, # min 공간 동작 카운터
          'park_laplacian_th': 3.5,       # park 라플라시안 wh
          'park_sec_to_wait': 5,          # park 기다리는 시간 초
          'start_frame': 0}  # 35000      # 시작 프레임

# Set capture device or file # 캡쳐도구와 파일을 설정한다.
cap = cv2.VideoCapture(fn) # 이 함수를 사용하여 비디오 캡쳐가 가능하다. (일반적으로 0 이면 기본 카메라) (안의 인자는 어떤 장치 인덱스를 사용할 것인지 지정, 여기서는 영상 경로인 fn을 지정한듯하다.)
video_info = {'fps': cap.get(cv2.CAP_PROP_FPS),                     # cap.get 을 사용하여 fps 정보를 가져온다. (=Frame rate.)
              'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),      # 너비 정보를 가져온다.
              'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),    # 높이 정보를 가져온다.
              'fourcc': cap.get(cv2.CAP_PROP_FOURCC),               # 포커스 정보를 가져온다. (코덱의 4문자? 4-character code of codec.
                                                                    # #cf(https://docs.opencv.org/3.4/d4/d15/group__videoio__flags__base.html#ggaeb8dd9c89c10a5c63c139bf7c4f5704da53e1c28d4c2ca10732af106f3bf00613)
              'num_of_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}  # 비디오 파일의 프레임 수를 가져온다.
cap.set(cv2.CAP_PROP_POS_FRAMES, config['start_frame'])  # jump to frame
# CAP_PROP_POS_FRAMES 프로퍼티의 값을 config['start_frame']로 지정하는 것이다.
# 위에서 CAP_PROP_POS_FRAMES는 0-based index of the frame to be decoded/captured next 이다.

# 코덱 정의 및 VideoWriter 개체 생성 #Define the codec and create VideoWriter object
# 동영상 재생시에는 해당 동영상의 Codec이 설치되어 있어야 한다.
if config['save_video']: # 비디오 저장시 : https://docs.opencv.org/3.1.0/dd/d43/tutorial_py_video_display.html
    # four cc 코드를  VideoWriter_fourcc에 전달함.
    fourcc = cv2.VideoWriter_fourcc('C', 'R', 'A',
                                    'M')  # options: ('P','I','M','1'), ('D','I','V','X'), ('M','J','P','G'), ('X','V','I','D')
    out = cv2.VideoWriter(fn_out, -1, 25.0,  # video_info['fps'],
                          (video_info['width'], video_info['height'])) # 저장될 비디오.

# HOG 설명자/사람 디텍터를 초기화합니다. #initialize the HOG descriptor/person detector
if config['pedestrian_detction']:
    #Histogram of Oriented Gradient
    hog = cv2.HOGDescriptor() # 관련 정보 링크 : https://blog.naver.com/tommybee/221173056260
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# 배경 감산기 작성 # Create Background subtractor
if config['motion_detection']:
    # 참고링크 : https://docs.opencv.org/3.4/de/de1/group__video__motion.html
    fgbg = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=16, detectShadows=True)
    # fgbg = cv2.createBackgroundSubtractorKNN(history=100, dist2Threshold=800.0, detectShadows=False)

# YAML데이터를 읽어온다. (주차공간의 사각형좌표) # Read YAML data (parking space polygons)
with open(fn_yaml, 'r') as stream: # 파일열기, fn_yaml를 읽기모드로 열었다. with문 밖을 벗어나면 자동으로 close를 해주기 때문에 사용한다. stream 이라는 이름의 변수로 파일을 사용가능하다.
    parking_data = yaml.load(stream)  # yaml.load() 로 yaml을 읽어온다..
parking_contours = [] # 파킹 윤곽
parking_bounding_rects = [] # 파킹 경계 사각형
parking_mask = []  # 파킹 마스크
for park in parking_data:  # 파킹 데이터에 있는 것을 반복
    points = np.array(park['points']) # park의 points 부분을 배열로 만든다. # np=numpy (NumPy관련 학습 必)
    rect = cv2.boundingRect(points) # 포인트 셋으로부터 사각형을 만든다.
    points_shifted = points.copy()
    points_shifted[:, 0] = points[:, 0] - rect[0]  # shift contour to roi
    points_shifted[:, 1] = points[:, 1] - rect[1]
    parking_contours.append(points)
    parking_bounding_rects.append(rect)
    mask = cv2.drawContours(np.zeros((rect[3], rect[2]), dtype=np.uint8), [points_shifted], contourIdx=-1,
                            color=255, thickness=-1, lineType=cv2.LINE_8)
    mask = mask == 255
    parking_mask.append(mask)

kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))  # morphological kernel
# kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(13,13)) # morphological kernel
kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 19))
parking_status = [False] * len(parking_data)
parking_buffer = [None] * len(parking_data)

while (cap.isOpened()):
    # Read frame-by-frame
    video_cur_pos = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0  # Current position of the video file in seconds
    video_cur_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)  # Index of the frame to be decoded/captured next
    ret, frame = cap.read()
    if ret == False:
        print("Capture Error")
        break

    # frame_gray = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2GRAY)
    # Background Subtraction
    frame_blur = cv2.GaussianBlur(frame.copy(), (5, 5), 3)
    frame_gray = cv2.cvtColor(frame_blur, cv2.COLOR_BGR2GRAY)
    frame_out = frame.copy()

    # Draw Overlay
    if config['text_overlay']:
        str_on_frame = "%d/%d" % (video_cur_frame, video_info['num_of_frames'])
        cv2.putText(frame_out, str_on_frame, (5, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 255), 2, cv2.LINE_AA)

    if config['motion_detection']:
        fgmask = fgbg.apply(frame_blur)
        bw = np.uint8(fgmask == 255) * 255
        bw = cv2.erode(bw, kernel_erode, iterations=1)
        bw = cv2.dilate(bw, kernel_dilate, iterations=1)
        (_, cnts, _) = cv2.findContours(bw.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # loop over the contours
        for c in cnts:
            # if the contour is too small, ignore it
            if cv2.contourArea(c) < config['min_area_motion_contour']:
                continue
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(frame_out, (x, y), (x + w, y + h), (255, 0, 0), 2)

    if config['parking_detection']:
        for ind, park in enumerate(parking_data):
            points = np.array(park['points'])
            rect = parking_bounding_rects[ind]
            roi_gray = frame_gray[rect[1]:(rect[1] + rect[3]),
                       rect[0]:(rect[0] + rect[2])]  # crop roi for faster calcluation
            laplacian = cv2.Laplacian(roi_gray, cv2.CV_64F)
            points[:, 0] = points[:, 0] - rect[0]  # shift contour to roi
            points[:, 1] = points[:, 1] - rect[1]
            delta = np.mean(np.abs(laplacian * parking_mask[ind]))
            status = delta < config['park_laplacian_th']
            # If detected a change in parking status, save the current time
            if status != parking_status[ind] and parking_buffer[ind] == None:
                parking_buffer[ind] = video_cur_pos
            # If status is still different than the one saved and counter is open
            elif status != parking_status[ind] and parking_buffer[ind] != None:
                if video_cur_pos - parking_buffer[ind] > config['park_sec_to_wait']:
                    parking_status[ind] = status
                    parking_buffer[ind] = None
            # If status is still same and counter is open
            elif status == parking_status[ind] and parking_buffer[ind] != None:
                # if video_cur_pos - parking_buffer[ind] > config['park_sec_to_wait']:
                parking_buffer[ind] = None
                # print("#%d: %.2f" % (ind, delta))
        # print(parking_status)

    if config['parking_overlay']:
        for ind, park in enumerate(parking_data):
            points = np.array(park['points'])
            if parking_status[ind]:
                color = (0, 255, 0)
            else:
                color = (0, 0, 255)
            cv2.drawContours(frame_out, [points], contourIdx=-1,
                             color=color, thickness=2, lineType=cv2.LINE_8)
            moments = cv2.moments(points)
            centroid = (int(moments['m10'] / moments['m00']) - 3, int(moments['m01'] / moments['m00']) + 3)
            cv2.putText(frame_out, str(park['id']), (centroid[0] + 1, centroid[1] + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame_out, str(park['id']), (centroid[0] - 1, centroid[1] - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame_out, str(park['id']), (centroid[0] + 1, centroid[1] - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame_out, str(park['id']), (centroid[0] - 1, centroid[1] + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame_out, str(park['id']), centroid, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    if config['pedestrian_detction']:
        # detect people in the image
        (rects, weights) = hog.detectMultiScale(frame, winStride=(4, 4), padding=(8, 8), scale=1.05)

        # draw the original bounding boxes
        for (x, y, w, h) in rects:
            cv2.rectangle(frame_out, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # write the output frame
    if config['save_video']:
        if video_cur_frame % 35 == 0:  # take every 30 frames
            out.write(frame_out)

            # Display video
    cv2.imshow('frame', frame_out)
    cv2.imshow('background mask', bw)
    k = cv2.waitKey(1)
    if k == ord('q'):
        break
    elif k == ord('c'):
        cv2.imwrite('frame%d.jpg' % video_cur_frame, frame_out)
    elif k == ord('j'):
        cap.set(cv2.CAP_PROP_POS_FRAMES, video_cur_frame + 10000)  # jump to frame

cap.release()
if config['save_video']: out.release()
cv2.destroyAllWindows()