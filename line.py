import cv2
import numpy as np
import tensorflow as tf


def pass_func(x):
   pass


snap_x = 200
i = 0

cv2.namedWindow('mask')
cv2.namedWindow('frame')

cap = cv2.VideoCapture('http://192.168.43.1:4747/video')

model = tf.keras.saving.load_model("model_local.keras")

cv2.createTrackbar('HL', 'mask', 0, 255, pass_func)
cv2.createTrackbar('SL', 'mask', 0, 255, pass_func)
cv2.createTrackbar('VL', 'mask', 239, 255, pass_func)
cv2.createTrackbar('HM', 'mask',205, 255, pass_func)
cv2.createTrackbar('SM', 'mask', 41, 255, pass_func)
cv2.createTrackbar('VM', 'mask', 255, 255, pass_func)

cv2.createTrackbar('dp', 'frame', 350, 1000, pass_func)
cv2.createTrackbar('minDist', 'frame', 16, 100, pass_func)
cv2.createTrackbar('param1', 'frame', 10, 500, pass_func)
cv2.createTrackbar('param2', 'frame', 2, 100, pass_func)
cv2.createTrackbar('minRadius', 'frame', 20, 100, pass_func)
cv2.createTrackbar('maxRadius', 'frame', 34, 200, pass_func)


while True:
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
   mask = frame.copy()
   hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
   mask = cv2.inRange(hsv, hsv_min, hsv_max)

   circles = cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT_ALT, 
                              dp = (c_dp/100), 
                              minDist = c_minDist, 
                              param1 = c_param1, 
                              param2 = (c_param2/100), 
                              minRadius = c_minRadius, 
                              maxRadius = c_maxRadius)

   if circles is not None:
      circles = np.round(circles[0, :]).astype("int")
      for (x, y, r) in circles:
         if abs(x - snap_x) <= 3:
            if y + 32 < frame.shape[0] and y - 32 > 0 and x + 32 < frame.shape[1] and x - 32 > 0:
               img = frame[y-32:y+32, x-32:x+32]
               img = np.expand_dims(img, axis = 0)
               predict = model.predict(img)
               if predict[0][0] > 0.9:
                  cv2.circle(frame, (x, y), r, (0, 255, 0), 4)
                  print(f'{i} good')
               else:
                  cv2.circle(frame, (x, y), r, (0, 0, 255), 4)   
                  print(f'{i} bad')
               i += 1

   cv2.line(frame, (snap_x, 0), (snap_x, frame.shape[1]), (255, 0, 0), 3)         

   cv2.imshow('frame', frame)
   cv2.imshow('mask', mask)

   if cv2.waitKey(1) & 0xFF == ord('q'):
      break

    
cap.release()
cv2.destroyAllWindows()