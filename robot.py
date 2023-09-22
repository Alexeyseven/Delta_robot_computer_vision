import socket


s = socket.socket()
s.connect(('127.0.0.1', 9090))

i = 0

while True:
    s.send(str.encode(str(i)))
    mes = s.recv(1024)
    i += 1

    if i == 1000:
        i = 0