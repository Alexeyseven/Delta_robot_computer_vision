import cv2
import numpy as np
import socket


cap = cv2.VideoCapture('http://192.168.43.1:4747/video')

s = socket.socket()
s.bind(('192.168.1.241', 9090))
s.listen(1)
conn, addr = s.accept()


while True:
    

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
