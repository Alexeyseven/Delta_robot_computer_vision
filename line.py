import cv2
import numpy as np
import os
import time
import tensorflow as tf
import socket
import threading
import datetime


model = tf.keras.saving.load_model("model.keras")

s = socket.socket()
s.bind(('192.168.1.241', 9090))
#s.bind(('192.168.1.241', 502))
s.listen(1)
os.system('start cmd /k python robot.py')
conn, addr = s.accept()
conn.settimeout(0.0001)


def pass_func(x):
   pass


def diapason_transpole(max_vis, min_vis, max_rob, min_rob, coordinate):
   return((max_rob-min_rob)*(coordinate-min_vis)/(max_vis-min_vis)+min_rob)


def prediction():
    global img
    global get_prediction
    global bad_count
    global coordinates
    global send_data
    global coordinates_detect

    while cap.isOpened():
        if get_prediction == True:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV_FULL)
            img = np.expand_dims(img, axis=0)
            coordinates = [x, y]
            predict = model.predict(img)
            if predict[0][0] == 1:
                bad_count += 1
                coordinates_detect = coordinates
                send_data = True
            get_prediction = False


def shift_register():
    global stack
    global mes
    
    while cap.isOpened():
        if len(stack) > 0:
            data = stack.pop(0)
            time_delta = data[1]
            if time_delta > shift:
                time_delta = shift
                conn.send(b'home\r\n')
            time.sleep(time_delta/1000)
            conn.send(str.encode(str(data[0])  + ',' + str(shift_y) + '\r\n'))


snap_y = 200
min_vis = 100
max_vis = 607
min_rob = 238
max_rob = 547
x_prev = 0
get_prediction = False
bad_count = 0
coordinates = []
send_data = False
coordinates_detect = [0, 0]
stack = []

past = datetime.datetime.now()

cv2.namedWindow('mask')
cv2.namedWindow('frame')

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"
cap = cv2.VideoCapture("video.avi")
#cap = cv2.VideoCapture('rtsp://admin:innotech@@192.168.1.64:554')

cv2.createTrackbar('HL', 'mask', 0, 255, pass_func)
cv2.createTrackbar('SL', 'mask', 100, 255, pass_func)
cv2.createTrackbar('VL', 'mask', 0, 255, pass_func)
cv2.createTrackbar('HM', 'mask', 255, 255, pass_func)
cv2.createTrackbar('SM', 'mask', 255, 255, pass_func)
cv2.createTrackbar('VM', 'mask', 255, 255, pass_func)

cv2.createTrackbar('shift', 'frame', 4570, 10000, pass_func)
cv2.createTrackbar('shift_y', 'frame', 100, 1000, pass_func)

thread1 = threading.Thread(target=prediction, name="Thread-1")
thread2 = threading.Thread(target=shift_register, name="Thread-2")
thread1.start()
thread2.start()


while True:
    mes = b'none\r\n'
    hl = cv2.getTrackbarPos('HL','mask')
    sl = cv2.getTrackbarPos('SL','mask')
    vl = cv2.getTrackbarPos('VL','mask')
    hm = cv2.getTrackbarPos('HM','mask')
    sm = cv2.getTrackbarPos('SM','mask')
    vm = cv2.getTrackbarPos('VM','mask')

    shift = cv2.getTrackbarPos('shift','frame')
    shift_y = cv2.getTrackbarPos('shift_y', 'frame')

    hsv_min = np.array((hl, sl, vl), np.uint8)
    hsv_max = np.array((hm, sm, vm), np.uint8)

    ret, frame = cap.read()
    frame = frame[0:570, 200:1100]
    mask = frame.copy()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, hsv_min, hsv_max)

    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    for cnt in contours:
        sm = cv2.arcLength(cnt, True)
        apd = cv2.approxPolyDP(cnt, 0.01*sm, True)
        area = cv2.contourArea(cnt)
        if area > 2700:
            if len(apd) > 8:
                [x,y,w,h] = cv2.boundingRect(cnt)
                if y < snap_y and abs(x - coordinates_detect[0]) < 10:
                    cv2.circle(frame, (int(x+w/2), int(y+h/2)), int(w/2), (0, 0, 255), 4)
                if abs(x - x_prev) > 10:
                    if abs(y - snap_y) < 10:
                        img = frame[y:y+64, x:x+64]
                        get_prediction = True
                        x_prev = x

    cv2.line(frame, (0, snap_y), (frame.shape[1], snap_y), (255, 0, 0), 3) 
    cv2.putText(frame, ('BAD ITEMS: ' + str(bad_count)), (20, 170), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow('frame', frame)
    cv2.imshow('mask', mask)

    if send_data == True:
        now = datetime.datetime.now()
        delta = now - past
        delta = delta.seconds* 1000 + int(delta.microseconds/1000)
        x = diapason_transpole(max_vis, min_vis, max_rob, min_rob, coordinates_detect[0])
        x = int(x)
        stack.append([x, delta])
        send_data = False
        past = now
    conn.send(mes)
    time.sleep(0.02)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    
cap.release()
cv2.destroyAllWindows()
thread1.join()
thread2.join()
