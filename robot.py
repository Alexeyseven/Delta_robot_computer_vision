import socket


i = 0

s = socket.socket()
s.connect(('127.0.0.1', 9090))
print('Robot ready')


while True:
    s.send(str.encode(str(i)))
    i += 1
    s.recv(50)
        