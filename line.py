import cv2
import numpy as np
import socket
import os
import time


def pass_func(x):
   pass

cv2.namedWindow('mask')

cap = cv2.VideoCapture('Video.mp4')
#cap = cv2.VideoCapture('http://192.168.43.1:4747/video')

cv2.createTrackbar('HL', 'mask', 0, 255, pass_func)
cv2.createTrackbar('SL', 'mask', 50, 255, pass_func)
cv2.createTrackbar('VL', 'mask', 150, 255, pass_func)
cv2.createTrackbar('HM', 'mask',70, 255, pass_func)
cv2.createTrackbar('SM', 'mask', 255, 255, pass_func)
cv2.createTrackbar('VM', 'mask', 255, 255, pass_func)

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

    contours, _ = cv2.findContours(mask, cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) > 500 and cv2.contourArea(cnt) < 1500:
            [x,y,w,h] = cv2.boundingRect(cnt)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            img = frame[y:y+45, x:x+45]
            cv2.imshow('img', img)

    cv2.imshow('segment', frame)
    cv2.imshow('mask', mask)
    time.sleep(0.1)
    
    conn.send(b'none')

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
