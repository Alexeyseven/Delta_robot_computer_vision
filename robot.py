import socket


s = socket.socket()
s.connect(('192.168.1.241', 9090))

i = 0
j = 0

while True:
    if i == j + 50:
        print('robot_move')
        s.send(b'robot_move')
    else:
        s.send(str.encode(f'{i}'))

    mes = s.recv(1024)

    if mes != b'none':
        print(mes)
        j = i

    i += 1
    if i == 1000:
        i = 0
        j = 0