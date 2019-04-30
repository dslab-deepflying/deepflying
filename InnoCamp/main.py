from __future__ import print_function, division
import keras
import matplotlib.pyplot as plt
import sys,os
import numpy as np

generator_path = 'GANS/models/dress.h5'


class Generator:
    def __init__(self):
        self.generator = keras.models.load_model(generator_path)
        self.generator.summary()

    def save_imgs(self):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, 100))
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        # Write these images to files
        # fig, axs = plt.subplots(r, c)
        # cnt = 0
        # for i in range(r):
        #     for j in range(c):
        #         axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
        #         axs[i, j].axis('off')
        #         cnt += 1
        # data_name = generator_path.split('/')[-1].split('.')[0]
        # fig_name = "dcgan_%s.png" % data_name
        # fig.savefig(fig_name)
        # plt.close()

        return gen_imgs


if __name__ == '__main__':
    generator = Generator()
    generator.save_imgs()

