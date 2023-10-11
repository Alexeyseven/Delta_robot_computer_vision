import socket


s = socket.socket()
s.connect(('127.0.0.1', 9090))
print('Robot ready')

i = 0
j = 0
delay = 5
y = 0
y_prev = 0

while True:
    if i == j + delay:
        s.send(b'robot move')
        if y != y_prev or delay > 0:
            if delay == 0:
                delay = 2
            print('robot move, y = ', y, 'delay = ', delay)
            y_prev = y
    else:
        s.send(str.encode(str(i)))
        
    i += 1

    if i == 1000:
        i = 0

    mes = s.recv(1024)

    if mes != b'none':
        line = mes.decode('utf-8').split(',')
        delay = int(line[0])
        y = int(line[1])
        j = i
        