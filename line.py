import cv2
import numpy as np
import socket


def pass_func(x):
   pass

cv2.namedWindow('frame')
cv2.namedWindow('mask')

x_detect = 100
stack = []
robot_move = False
j = 0

#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('http://192.168.43.1:4747/video')

cv2.createTrackbar('HL', 'mask', 11, 255, pass_func)
cv2.createTrackbar('SL', 'mask', 73, 255, pass_func)
cv2.createTrackbar('VL', 'mask', 173, 255, pass_func)
cv2.createTrackbar('HM', 'mask',23, 255, pass_func)
cv2.createTrackbar('SM', 'mask', 255, 255, pass_func)
cv2.createTrackbar('VM', 'mask', 255, 255, pass_func)

s = socket.socket()
s.bind(('192.168.1.241', 9090))
s.listen(1)
conn, addr = s.accept()

while True:
    mes = conn.recv(1024)

    hl = cv2.getTrackbarPos('HL','mask')
    sl = cv2.getTrackbarPos('SL','mask')
    vl = cv2.getTrackbarPos('VL','mask')
    hm = cv2.getTrackbarPos('HM','mask')
    sm = cv2.getTrackbarPos('SM','mask')
    vm = cv2.getTrackbarPos('VM','mask')

    hsv_min = np.array((hl, sl, vl), np.uint8)
    hsv_max = np.array((hm, sm, vm), np.uint8)
   
    ret, frame = cap.read()
    flag, track = cap.read()

    cropped = frame[0:480, 0:480]
    hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, hsv_min, hsv_max)

    cv2.line(frame, (x_detect, 0), (x_detect, frame.shape[1]), (255, 0, 0), 3)
    
    if mes == b'robot_move':
        print(mes)
        robot_move = True
    else:
        i = int(mes.decode('utf-8'))    
            
    if len(stack)>0 and robot_move==True:
        translation = stack.pop(0)
        translation = str(translation[0]) + ', ' + str(translation[1])
        conn.send(str.encode(translation))
        robot_move = False


    contours, _ = cv2.findContours(mask, cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt)>5000:
            [x,y,w,h] = cv2.boundingRect(cnt)
            if x == x_detect:
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
                cv2.imshow('object', frame[y:y+128, x:x+128])
                delay = i-j
                if delay > 50:
                    delay = 50
                j = i
                stack.append([y, delay])
            else:
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

    cv2.imshow('frame', frame)
    cv2.imshow('mask', mask)
    
    conn.send(b'none')

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
