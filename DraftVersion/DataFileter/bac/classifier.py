#!/usr/bin/python
from scipy.spatial import distance
from scipy.cluster import hierarchy
from keras.applications.xception import Xception
from keras.preprocessing import image
from keras.applications.xception import preprocess_input
from keras.models import Model
import numpy as np
import pandas as pd
import sys,os,shutil
import re, pickle
import PIL.Image
pj = os.path.join


def read_pk(fn):
    with open(fn, 'rb') as fd:
        ret = pickle.load(fd)
    return ret


def write_pk(obj, fn):
    with open(fn, 'wb') as fd:
        pickle.dump(obj, fd)


def get_files(dr, ext='jpg|jpeg|bmp|png'):
    rex = re.compile(r'^.*\.({})$'.format(ext), re.I)
    return [os.path.join(dr,base) for base in os.listdir(dr) if rex.match(base)]


def get_model():

    base_model = Xception(weights='imagenet', include_top=True)
    model = Model(inputs=base_model.input,
                  outputs=base_model.get_layer('avg_pool').output)
    return model


def fingerprint(fn, model, size):
    """Load image from file `fn`, resize to `size` and run through `model`
    (keras.models.Model).

    Parameters
    ----------
    fn : str
        filename
    model : keras.models.Model instance
    size : tuple
        input image size (width, height), must match `model`, e.g. (224,224)

    Returns
    -------
    fingerprint : 1d array
    """
    # print(fn)

    # keras.preprocessing.image.load_img() uses img.rezize(shape) with the
    # default interpolation of PIL.Image.resize() which is pretty bad (see
    # imagecluster/play/pil_resample_methods.py). Given that we are restricted
    # to small inputs of 224x224 by the VGG network, we should do our best to
    # keep as much information from the original image as possible. This is a
    # gut feeling, untested. But given that model.predict() is 10x slower than
    # PIL image loading and resizing .. who cares.
    #
    # (224, 224, 3)
    ##img = image.load_img(fn, target_size=size)
    img = PIL.Image.open(fn).resize(size, 3)

    # (224, 224, {3,1})
    arr3d = image.img_to_array(img)

    # (224, 224, 1) -> (224, 224, 3)
    #
    # Simple hack to convert a grayscale image to fake RGB by replication of
    # the image data to all 3 channels.
    #
    # Deep learning models may have learned color-specific filters, but the
    # assumption is that structural image features (edges etc) contibute more to
    # the image representation than color, such that this hack makes it possible
    # to process gray-scale images with nets trained on color images (like
    # VGG16).
    if arr3d.shape[2] == 1:
        arr3d = arr3d.repeat(3, axis=2)

    # Indeed, this gray-scale-hack code does not work at all
    # You will find they have no difference between source-image
    # when you display them after that
    # PIL.Image._show(image.array_to_img(arr3d))

    # (1, 224, 224, 3)
    arr4d = np.expand_dims(arr3d, axis=0)

    # (1, 224, 224, 3)
    arr4d_pp = preprocess_input(arr4d)

    # Original code of return all of this array
    return model.predict(arr4d_pp)[0, :]

    # use prediction of cloth to be a fingerprint
    # however it has poor ability to cluster them ...

    # cates = pd.read_table(cate_index_txt_path,sep=',')['index']
    # arr_p = model.predict(arr4d_pp)[0,:]
    # arr_p = [arr_p[i] for i in cates]
    #
    # return arr_p

imgs_folder_path = ''
fp_folder_path = ''
