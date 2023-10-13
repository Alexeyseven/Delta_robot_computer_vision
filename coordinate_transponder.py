min_vis = 70
max_vis = 340
min_rob = -130
max_rob = -480

x = 80


res = (max_rob-min_rob)*(x-min_vis)/(max_vis-min_vis)+min_rob
print(int(res))