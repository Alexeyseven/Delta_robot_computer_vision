import cv2
import numpy as np
import socket
import os
import time


cap = cv2.VideoCapture('Движение продукта на конвейере.mp4')
#cap = cv2.VideoCapture('http://192.168.43.1:4747/video')

s = socket.socket()
s.bind(('127.0.0.1', 9090))
s.listen(1)
os.system('start cmd /k python robot.py')
conn, addr = s.accept()

j = 0


while True:
    mes = conn.recv(1024)
    i = int(mes.decode('utf-8'))

    ret, frame = cap.read()

    final_wide = 500
    r = float(final_wide) / frame.shape[1]
    dim = (final_wide, int(frame.shape[0] * r))
    frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
    frame = frame[200:590, 0:200]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow('segment', frame)
    time.sleep(0.1)
    
    conn.send(b'none')

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
