from keras.utils import to_categorical
from keras import layers
from keras import models
import matplotlib.pyplot as plot
import numpy as np
import os
import cv2

train_labels = []

path1 = 'Data_good'
path2 = 'Data_bad'

train_images = cv2.imread(f'{path1}/{os.listdir(path1)[0]}')
train_images = cv2.cvtColor(train_images, cv2.COLOR_BGR2GRAY)
train_images = np.expand_dims(train_images, axis=0)
print(train_images.shape)


for i in os.listdir(path1):
    img = cv2.imread(f'{path1}/{i}')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = np.expand_dims(img, axis=0)
    train_images = np.append(train_images, img, axis=0)
    train_labels = np.append(train_labels, 1)
train_images = train_images[1:]
print(train_images.shape)


for i in os.listdir(path2):
    img = cv2.imread(f'{path2}/{i}')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = np.expand_dims(img, axis=0)
    train_images = np.append(train_images, img, axis=0)
    train_labels = np.append(train_labels, 0)
print(train_images.shape)

train_labels = np.array(train_labels)
print(train_labels.shape)

train_images = train_images.reshape((3238, 48, 48, 1))
train_images = train_images.astype('float32') / 255
train_labels = to_categorical(train_labels)
print(train_images.shape, train_labels.shape)

model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation='relu',
                        input_shape=(48, 48, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(2, activation='softmax'))
model.summary()

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=10, batch_size=20)

#model.evaluate(test_images, test_labels)

model.save('myaphly_model_local.keras')
input('Done\n')