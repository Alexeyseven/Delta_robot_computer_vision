import socket
import os
from tkinter import *


message = False

s = socket.socket()
s.bind(('192.168.1.241', 502))
s.listen(1)
#os.system('start cmd /k python socket_client.py')
conn, addr = s.accept()
conn.settimeout(0.05)


def send_message():
    global message
    message = True


root = Tk()
Button(text='send_message', command=send_message).place(x=10, y=10)

while True:
    try:
        data = conn.recv(1024)
        print(data)
    except socket.timeout:
        if message:
            data = b'404\r\n'
        else:
            data = b'none\r\n'
        conn.send(data)   
        message = False  
        root.update()