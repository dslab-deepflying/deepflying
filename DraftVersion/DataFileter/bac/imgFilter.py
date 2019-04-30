#! /usr/bin/python
import numpy as np
from PIL import Image
from PIL import ImageFilter
import matplotlib.pyplot as plt

from skimage import measure,draw,morphology,color

test_img_path = '/home/jc/deepcam/Data/categories/Jeans/Bleached_Skinny_Jeans/img_00000005.jpg'

def get_binary_img(img_path,threshold = 200):

    """
    :param img_path:
    :param threshold: for converting binary img
    :return:numpy array with 0 and 1
    """
    img = Image.open(img_path)
    img.show()


    # level image
    img = img.convert("L")

    # binary image
    WHITE, BLACK = 255, 0
    img = img.point(lambda x: WHITE if x > threshold else BLACK)
    img = img.convert('1')

    # use filter to remove some noise
    # (such like text)
    img = img.filter(ImageFilter.BLUR)
    img = img.filter(ImageFilter.BLUR)
    img = img.filter(ImageFilter.BLUR)

    #img.show()
    # print np.array(img).shape

    return np.array(img)

def get_outer(bin_img_arr):

#///
    bin_img_arr = morphology.remove_small_objects(bin_img_arr, min_size=7999, connectivity=2)

    contours = measure.find_contours(bin_img_arr, 0.5)

    labels=measure.label(bin_img_arr,connectivity=2)
    dst=color.label2rgb(labels)
    # print('regions number:',labels.max()+1)

    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(8, 12))
    ax0.imshow(bin_img_arr, plt.cm.gray)
    ax1.imshow(bin_img_arr, plt.cm.gray)
    ax2.imshow(dst,interpolation='nearest')
    ax2.axis('off')
    for n, contour in enumerate(contours):
        ax1.plot(contour[:, 1], contour[:, 0], linewidth=2)
    ax1.axis('image')
    ax1.set_xticks([])
    ax1.set_yticks([])
    plt.show()

def get_noskin_img(img_path):

    def is_skin(data):
        if (data[0] >= 180 and data[0] <= 245 and
            data[1] >= 170 and data[1] <= 230 and
            data[2] >= 170 and data[2] <= 210):
            print ('true')
            return True

        return False

    img = Image.open(img_path)

    img.show()

    width = img.size[0]
    height = img.size[1]

    for i in range(width):
        for j in range(height):
            data = (img.getpixel((i,j)))
            if is_skin(data):
                img.putpixel((i,j),(0,0,0))

    img = img.convert("RGB")



    img.show()




if __name__ == '__main__':
    # img = get_binary_img(test_img_path)
    # get_outer(img)

    get_noskin_img(test_img_path)

