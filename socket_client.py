import socket
from tkinter import *


s = socket.socket()
s.connect(('192.168.1.241',9090))


def send():
    s.send(b'send_data')


root = Tk()
Button(text='send', command=send).place(x=10, y=10)


while True:
    root.update()