import socket


s = socket.socket()
s.connect(('127.0.0.1', 9090))
print('Robot ready')

i = 0
j = 0

while True:
    if i == (j + 5):
        s.send(b'robot move')
        print('robot move')
    else:
        s.send(str.encode(str(i)))
        
    i += 1

    if i == 1000:
        i = 0

    mes = s.recv(1024)

    if mes != b'none':
        print(mes)
        j = i
        