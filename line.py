import cv2
import numpy as np
import socket
import os
import time


def pass_func(x):
   pass


cv2.namedWindow('frame')
cv2.namedWindow('mask')

cap = cv2.VideoCapture('Движение продукта на конвейере.mp4')
#cap = cv2.VideoCapture('http://192.168.43.1:4747/video')

cv2.createTrackbar('HL', 'mask', 0, 255, pass_func)
cv2.createTrackbar('SL', 'mask', 104, 255, pass_func)
cv2.createTrackbar('VL', 'mask', 129, 255, pass_func)
cv2.createTrackbar('HM', 'mask',71, 255, pass_func)
cv2.createTrackbar('SM', 'mask', 255, 255, pass_func)
cv2.createTrackbar('VM', 'mask', 255, 255, pass_func)

s = socket.socket()
s.bind(('127.0.0.1', 9090))
s.listen(1)
os.system('start cmd /k python robot.py')
conn, addr = s.accept()


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
    time.sleep(0.1)

    final_wide = 500
    r = float(final_wide) / frame.shape[1]
    dim = (final_wide, int(frame.shape[0] * r))
    frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
    track = frame.copy()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, hsv_min, hsv_max)
    segment = frame[0:, 0:200]

    contours, _ = cv2.findContours(mask, cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt)>500 and cv2.contourArea(cnt)<2000:
            [x,y,w,h] = cv2.boundingRect(cnt)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

    cv2.imshow('segment', segment)
    cv2.imshow('mask', mask)
    
    conn.send(b'none')

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
