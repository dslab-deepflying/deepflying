#!/usr/bin/python
#-*- coding=utf-8 -*-
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import sys
from keras.preprocessing.image import load_img,img_to_array,save_img
from keras.applications import vgg19
import  numpy as np
from keras import backend as K
from scipy.optimize import fmin_l_bfgs_b

import time

target_image_path = 'D:\codes\PyCharm\deepflying\InnoCamp\SF\pics\content.png'
style_reference_path = 'D:\codes\PyCharm\deepflying\InnoCamp\SF\pics\style.jpg'
img_height = 200
img_width = 200
iter_size = 1


def preprocess_image(image_path):
    img = load_img(image_path,target_size=(img_height,img_width))
    img = img_to_array(img)
    img = np.expand_dims(img,axis=0)
    img = vgg19.preprocess_input(img)
    return img

def deprocess_image(x):
    x[:,:,0] += 103.939
    x[:,:,1] += 116.779
    x[:,:,2] += 123.68
    x = x[:,:,::-1] # BGR ---> RGB
    x = np.clip(x,0,255).astype('uint8')
    return x

def content_loss(base,combination):
    return  K.sum(K.square(combination- base))

def gram_matrix(x):
    features = K.batch_flatten(K.permute_dimensions(x,(2,0,1)))
    gram = K.dot(features,K.transpose(features))
    return gram

def style_loss(style,combination):
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    size = img_height * img_width
    return  K.sum(K.square(S-C) )/ (4.*(channels**2) *(size**2))

def total_variation_loss(x):
    a= K.square(
        x[: , :img_height - 1,:img_width-1,:] -
        x[: ,1:              ,:img_width-1,:]
    )
    b = K.square(
        x[:,:img_height - 1, :img_width - 1,:] -
        x[:,:img_height - 1,1:             ,:]
    )
    return K.sum(K.pow(a+b,1.25))



def main():
    width, height = load_img(target_image_path).size
    global img_height,img_width
    img_height = 200
    img_width = int(width * img_height / height)


    target_image = K.constant(preprocess_image(target_image_path))
    style_reference_image = K.constant(preprocess_image(style_reference_path))
    combination_image = K.placeholder((1,img_height,img_width,3))
    input_tensor = K.concatenate([target_image, style_reference_image,
                                  combination_image], axis=0)
    model = vgg19.VGG19(input_tensor= input_tensor,weights='imagenet',
                        include_top=False)

    output_dict = dict([(layer.name,layer.output) for layer in model.layers ])
    content_layer = 'block5_conv2'
    style_layers = ['block1_conv1', 'block2_conv1',
                    'block3_conv1', 'block4_conv1', 'block5_conv1']

    total_variation_weight = 1e-4
    style_weight = 1.
    content_weight = 0.025

    loss = K.variable(0.)
    layer_features = output_dict[content_layer]
    target_image_features = layer_features[0,: ,:,:]
    combination_features =  layer_features[2,: ,:,:]
    loss += content_weight*content_loss(target_image_features,combination_features)

    for layer_name in style_layers:
        layer_features = output_dict[layer_name]
        style_reference_features = layer_features[1,:,:,:]
        combination_features =     layer_features[2,:,:,:]
        s1 = style_loss(style_reference_features,combination_features)
        loss += (style_weight/len(style_layers))*s1

    loss += total_variation_weight * total_variation_loss(combination_image)

    grads = K.gradients(loss, combination_image)[0]
    fetch_loss_and_grads = K.function([combination_image], [loss, grads])

    class Evaluator(object):
        def __init__(self):
            self.loss_value = None
            self.grads_values = None

        def loss(self, x):
            assert self.loss_value is None
            x = x.reshape((1, img_height, img_width, 3))
            outs = fetch_loss_and_grads([x])
            loss_value = outs[0]
            grad_values = outs[1].flatten().astype('float64')
            self.loss_value = loss_value
            self.grad_values = grad_values
            return self.loss_value

        def grads(self, x):
            assert self.loss_value is not None
            grad_values = np.copy(self.grad_values)
            self.loss_value = None
            self.grad_values = None
            return grad_values

    evaluator = Evaluator()

    iterations = iter_size

    x = preprocess_image(target_image_path)
    x = x.flatten()

    for i in range(iterations):

        x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x,
                                         fprime=evaluator.grads, maxfun=20)
        img = x.copy().reshape((img_height, img_width, 3))
        img = deprocess_image(img)
        percent = (int)(100.0*i/iterations)
        sys.stdout.write("\r{0}{1}{2}{3}{4}".format("\r[%2d%%]" % percent, "[", "="*int(percent/5), " "*(20-int(percent/5)), "]"))
        sys.stdout.flush()
        if(i == iterations-1):
            fname = 'NST.png'
            save_img(fname, img)
            print('\n\rImage saved as %s \n\r'% fname)
    print('\n\r')




if __name__ == 'main' or __name__ == '__main__':
        main()
