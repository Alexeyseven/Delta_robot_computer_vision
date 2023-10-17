import cv2
import numpy as np
import socket
import os
import tensorflow as tf
import random


def pass_func(x):
   pass


def transport_start():
   pass


def transport_stop():
   pass


def diapason_transpole(max_vis, min_vis, max_rob, min_rob, coordinate):
   return((max_rob-min_rob)*(coordinate-min_vis)/(max_vis-min_vis)+min_rob)


cv2.namedWindow('mask')
cv2.namedWindow('frame')

model = tf.keras.saving.load_model("myaphly_model_local.keras")

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
cv2.createTrackbar('maxRadius', 'frame', 28, 200, pass_func)

s = socket.socket()
s.bind(('192.168.1.241', 502))
s.listen(1)
os.system('start cmd /k python robot.py')
conn, addr = s.accept()
conn.settimeout(0.05)

snap = False
stack = []
min_vis = 0
max_vis = 400
min_rob = -480
max_rob = -130


while True:
   try:
      mes = conn.recv(1024)
      if mes == b'snap\r\n':
         snap = True
         print(snap)
      else:
         print('robot say', mes)   
   except socket.timeout: 

      if len(stack) > 0:
         send_mes = stack.pop(0)
         send_mes = str.encode(str(send_mes[0]) + ',' + str(send_mes[1]) + '\r\n')
      else:
         send_mes = b'none\r\n'


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

      #cv2.line(frame, (snap_x, 0), (snap_x, frame.shape[1]), (255, 0, 0), 3)

      circles = cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT_ALT, 
                                 dp = (c_dp/100), 
                                 minDist = c_minDist, 
                                 param1 = c_param1, 
                                 param2 = (c_param2/100), 
                                 minRadius = c_minRadius, 
                                 maxRadius = c_maxRadius)

      if circles is not None:
         circles = np.round(circles[0, :]).astype("int")
         r2 = random.randint(-12, 12)
         if snap:
            for (x, y, r) in circles:
               if y + 24 < frame.shape[0] and y - 24 > 0 and x + 24 < frame.shape[1] and x -24 > 0:
                  img = frame[y-24:y+24, x-24:x+24]
                  r1 = random.randint(0, 50)
                  if r1 == 4:
                     cv2.circle(frame, (x-r2, y-r2), 4, (0, 0, 0), 1)
                  img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                  img = np.expand_dims(img, axis = 0)
                  cv2.circle(frame, (x, y), r, (255, 0, 0), 4)
                  predict = model.predict(img)
                  if predict[0][0] > 0.5:
                     cv2.circle(frame, (x, y), r, (0, 0, 255), 4)
                     stack.append([x, y])
                  else:
                     cv2.circle(frame, (x, y), r, (0, 255, 0), 4)
            snap = False

      cv2.imshow('frame', frame)
      cv2.imshow('mask', mask)
      
      conn.send(send_mes)


   if cv2.waitKey(1) & 0xFF == ord('q'):
      break

    
cap.release()
cv2.destroyAllWindows()