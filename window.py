from tkinter import *


def transport_start():
   pass


def transport_stop():
   pass


root = Tk()
root.geometry('700x40+0+700')
Button(text='Транспорт старт', command=transport_start, height=2).place(x=0, y=0)
Button(text='Транспорт стоп', command=transport_stop, height=2).place(x=100, y=0)


while True:
   root.update()