#! /usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
import os,sys
import keras


latent_dim = 100
model_name = 'models/wgan_240_240_50001_Jeans_G.h5'

def model_test():
    r, c = 5, 5

    model = keras.models.load_model(model_name)
    model.summary()

    noise = np.random.normal(0, 1, (r * c, latent_dim))
    gen_imgs = model.predict(noise)

    # Rescale images 0 - 1
    gen_imgs = 0.5 * gen_imgs + 0.5

    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i,j].imshow(gen_imgs[cnt, :,:,:])
            axs[i,j].axis('off')
            cnt += 1
    save_path = 'images/Check'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    fig_name = "%s.png" % (model_name.split('/')[1].split('.')[0])
    print (fig_name)
    if os.path.exists(sys.path[0]+"/"+fig_name):
        os.remove(sys.path[0]+"/"+fig_name)
    fig.savefig(save_path+'/'+fig_name)
    plt.close()

if __name__ == "__main__":
    model_test()