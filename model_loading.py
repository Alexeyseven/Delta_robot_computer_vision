import matplotlib.pyplot as plot
import numpy as np
import os
import cv2
from tkinter import *
import tensorflow as tf

i = 0
path = 'test_images/'

#model = models.load_model('myaphly_model_local.h5')
model = tf.keras.saving.load_model("model_local.keras")


def predict():
    global i
    j = os.listdir(path)[i]
    img = cv2.imread(f'{path}{j}')
    plot.imshow(img, cmap=plot.cm.binary)
    plot.show()
    img = np.expand_dims(img, axis = 0)
    predict = model.predict(img)
    print(predict)
    i += 1


root = Tk()
Button(text='predict', command = predict).place(x=10, y=10)
input('Done\n')