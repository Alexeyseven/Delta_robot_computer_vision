import os
import cv2


j = 0


for i in os.listdir('train_images'):
    print(i, j)
    frame = cv2.imread(f'train_images/{i}')
    cv2.imwrite(f'train_images/{j}.jpg', frame)
    j+=1
