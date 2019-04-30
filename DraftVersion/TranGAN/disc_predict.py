#!/usr/bin/python

import keras
import pandas as pd
import numpy as np
import random
from PIL import Image

model_path = 'models/dcgan_Dtee_50001.h5'
data_path = 'data/tee.csv'

test_img_path = '/home/deepcam/Data/categories/Tee/Bad_Influence_Tee/img_00000036.jpg'

def get_ori_data():
    data = pd.read_csv(data_path, header=None)
    X_train = []

    for i in range(1, len(data)):
        img = np.uint8(np.array(data.iloc[i]).reshape(28, 28, 1))
        X_train.append(img)

    X_train = random.sample(X_train, 100)
    X_train = np.array(X_train)

    # Rescale -1 to 1
    X_train = X_train / 127.5 - 1.

    return  X_train

def get_real_data():
    img = Image.open(test_img_path)
    img = img.convert("L")

    img = img.resize((28, 28))
    width = img.size[0]
    height = img.size[1]

    print img.size
    for i in range(width):
        for j in range(height):
            img.putpixel((i,j),(255 - img.getpixel((i, j))))

    img.show()

    arr = np.asarray(img)
    arr = arr.reshape(1, 28, 28, 1)
    print arr.shape
    return arr


def main():
    model = keras.models.load_model(model_path)
    model .summary()

    X_train = get_real_data()
    res = model.predict(X_train)
    res = np.array(res).reshape(-1)

    print res
    print np.mean(res)
    print np.var(res)




main()