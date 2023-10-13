import socket


s = socket.socket()
s.connect(('192.168.1.241', 502))
print('Robot ready')

i = 0
j = 0
delay = 5
y = 0
y_prev = 0

while True:
    if i == j + delay:
        s.send(b'robot move')
        if y != y_prev:
            print('robot move, y = ', y,)
            y_prev = y
        
    i += 1

    if i == 1000:
        i = 0

    mes = s.recv(1024)

    if mes != b'none\r\n':
        line = mes.decode('utf-8')
        y = int(line)
        j = i
        