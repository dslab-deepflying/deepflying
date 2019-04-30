#!/usr/bin/python

from PIL import Image
from PIL import ImageFilter
import numpy as np
from sklearn.cluster import KMeans
import sklearn.cluster as skc
from sklearn import metrics

test_img_path = '/home/jc/deepcam/Data/categories/Jeans/Bleached_Skinny_Jeans/img_00000005.jpg'
img_rows = 128
img_cols = 128
img_chanels = 3
def main():
    km = KMeans(n_clusters=5)
    data,img = get_data()
    label = km.fit_predict(data)
    for i in range(img_rows):
        for j in range(img_cols):
            img.putpixel((j,i),(0,label[i*img_rows+j]*80,0))
    img = img.resize((400,400))
    img.show()
    img.save("%dx%d_%s.png"%(img_rows,img_cols,test_img_path.split('/')[-1].split('.')[0]))


def get_data():
    img = Image.open(test_img_path)
    img = img.resize((img_rows, img_cols))
    arr = np.asarray(img).reshape((img_rows*img_cols,img_chanels))


    return arr,img

main()