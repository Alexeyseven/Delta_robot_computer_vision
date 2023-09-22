import matplotlib.pyplot as plot
import numpy as np
import os
import cv2
from tkinter import *
import tensorflow as tf

i = 0
#model = models.load_model('myaphly_model_local.h5')
model = tf.keras.saving.load_model("myaphly_model_local.keras")


def predict():
    global i
    j = os.listdir('test_images/')[i]
    img = cv2.imread(f'test_images/{j}')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    plot.imshow(img, cmap=plot.cm.binary)
    plot.show()
    img = np.expand_dims(img, axis = 0)
    predict = model.predict(img)
    print(predict)
    i += 1


root = Tk()
Button(text='predict', command = predict).place(x=10, y=10)
