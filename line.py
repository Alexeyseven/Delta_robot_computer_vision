import cv2
import numpy as np
import socket
import os
import time
import random


def pass_func(x):
   pass

cv2.namedWindow('mask')
cv2.namedWindow('frame')


cap = cv2.VideoCapture('Video.mp4')
#cap = cv2.VideoCapture('http://192.168.43.1:4747/video')

cv2.createTrackbar('HL', 'mask', 0, 255, pass_func)
cv2.createTrackbar('SL', 'mask', 50, 255, pass_func)
cv2.createTrackbar('VL', 'mask', 150, 255, pass_func)
cv2.createTrackbar('HM', 'mask',70, 255, pass_func)
cv2.createTrackbar('SM', 'mask', 255, 255, pass_func)
cv2.createTrackbar('VM', 'mask', 255, 255, pass_func)

cv2.createTrackbar('dp', 'frame', 350, 1000, pass_func)
cv2.createTrackbar('minDist', 'frame', 16, 100, pass_func)
cv2.createTrackbar('param1', 'frame', 10, 500, pass_func)
cv2.createTrackbar('param2', 'frame', 2, 100, pass_func)
cv2.createTrackbar('minRadius', 'frame', 18, 100, pass_func)
cv2.createTrackbar('maxRadius', 'frame', 24, 200, pass_func)

s = socket.socket()
s.bind(('127.0.0.1', 9090))
s.listen(1)
os.system('start cmd /k python robot.py')
conn, addr = s.accept()

j = 0


while True:
    mes = conn.recv(1024)
    i = int(mes.decode('utf-8'))

    hl = cv2.getTrackbarPos('HL','mask')
    sl = cv2.getTrackbarPos('SL','mask')
    vl = cv2.getTrackbarPos('VL','mask')
    hm = cv2.getTrackbarPos('HM','mask')
    sm = cv2.getTrackbarPos('SM','mask')
    vm = cv2.getTrackbarPos('VM','mask')

    c_dp = cv2.getTrackbarPos('dp','frame')
    c_minDist = cv2.getTrackbarPos('minDist','frame')
    c_param1 = cv2.getTrackbarPos('param1','frame')
    c_param2 = cv2.getTrackbarPos('param2','frame')
    c_minRadius = cv2.getTrackbarPos('minRadius','frame')
    c_maxRadius = cv2.getTrackbarPos('maxRadius','frame')

    hsv_min = np.array((hl, sl, vl), np.uint8)
    hsv_max = np.array((hm, sm, vm), np.uint8)

    ret, frame = cap.read()

    final_wide = 500
    r = float(final_wide) / frame.shape[1]
    dim = (final_wide, int(frame.shape[0] * r))
    frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
    frame = frame[200:590, 0:500]
    mask = frame.copy()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, hsv_min, hsv_max)

    circles = cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT_ALT, dp = (c_dp/100), minDist = c_minDist, param1 = c_param1, 
                            param2 = (c_param2/100), minRadius = c_minRadius, maxRadius = c_maxRadius)

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")

        for (x, y, r) in circles:
            r1 = random.randint(-12, 12)
            r2 = random.randint(0, 1)
            if r2 == 0:
               cv2.circle(frame, (x, y), r, (0, 0, 0), 1)
            if y + 24 < frame.shape[0] and y - 24 > 0 and x + 24 < frame.shape[1] and x -24 > 0:
                if r2 == 1:
                  cv2.circle(frame, (x-r1, y-r1), r-18, (0, 0, 0), 1)
                #cv2.imshow('obj', frame[y-24:y+24, x-24:x+24])
                cv2.imwrite(f'train_images/bad/{j}.jpg', frame[y-24:y+24, x-24:x+24])
                j += 1
            
            
    cv2.imshow('frame', frame)
    cv2.imshow('mask', mask)
    time.sleep(0.1)
    
    conn.send(b'none')

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
