import socket
from tkinter import *


translation = b'none'


def send_mes():
    global translation
    translation = b'snap\r\n'


s = socket.socket()
s.connect(('192.168.1.241', 502))
print('Robot ready')

root = Tk()
Button(text='snap', command=send_mes).place(x=10, y=0)


while True:
    if translation != b'none\r\n':
        s.send(translation)
        translation = b'none\r\n'
    root.update()    