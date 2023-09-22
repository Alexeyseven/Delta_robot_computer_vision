import cv2
import numpy as np
import socket
import os


#cap = cv2.VideoCapture('http://192.168.43.1:4747/video')
cap = cv2.VideoCapture(0)

s = socket.socket()
s.bind(('127.0.0.1', 9090))
s.listen(1)
os.system('start cmd /k python robot.py')
conn, addr = s.accept()


while True:
    mes = conn.recv(1024)
    conn.send(b'ok')

    

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
