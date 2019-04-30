from __future__ import print_function, division

from keras.datasets import mnist
from keras import backend as K, Sequential
from keras.layers import Input, Dense, Reshape, Flatten, Lambda,Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D,BatchNormalization,Reshape
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D,Conv2DTranspose
from keras.models import Sequential, Model
from keras.optimizers import Adam

import matplotlib.pyplot as plt

import sys

import numpy as np

class GAN():

    def __init__(self):
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 256

        optimizer = Adam(0.0005, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer = optimizer,
            metrics = ['accuracy'])

        # Build the encoder
        self.encoder = self.build_encoder()

        # Build the generator
        self.generator = self.build_generator()

        # Build the Variational Auto-encoder
        self.vae = self.build_vae()
        self.vae.compile(loss='binary_crossentropy', optimizer=optimizer)

        # For the combined model we will only train the generator
        self.discriminator.trainable = True

    def build_encoder(self):

        model = Sequential(name='encoder')

        # model.add(Conv2D(64, kernel_size=2, strides=2, padding="same"))
        # model.add(Conv2D(128, kernel_size=2, strides=2, padding="same"))
        # model.add(Conv2D(256, kernel_size=2, strides=2, padding="same"))
        model.add(Flatten())
        model.add(Dense(256,kernel_regularizer='l2',kernel_initializer='he_uniform'))
        # model.add(BatchNormalization())
        model.add(LeakyReLU())

        return model

    def build_generator(self):
        """
        Build decoder/generator
        :return:
        """
        model = Sequential(name='decoder/generator')

        model.add(Dense(256, input_dim=self.latent_dim,kernel_regularizer='l2',kernel_initializer='he_uniform'))
        model.add(LeakyReLU(alpha=0.2))
        # model.add(Reshape([2,2,64]))
        # model.add(BatchNormalization(momentum=0.8))
        # model.add(Conv2DTranspose(256, kernel_size=3))
        # model.add(LeakyReLU(alpha=0.2))
        # model.add(BatchNormalization(momentum=0.8))
        # model.add(Conv2DTranspose(128, kernel_size=3))
        # model.add(LeakyReLU(alpha=0.2))
        # model.add(BatchNormalization(momentum=0.8))
        # model.add(Flatten())
        model.add(Dense(np.prod(self.img_shape), activation='tanh'))
        model.add(Reshape(self.img_shape))

        return model

    def build_vae(self):

        x = Input(shape=self.img_shape)

        h = self.encoder(x)

        z_mean = Dense(self.latent_dim)(h)
        z_log_var = Dense(self.latent_dim)(h)

        def sampling(args):
            z_mean, z_log_var = args
            epsilon = K.random_normal(shape=(K.shape(z_mean)[0], self.latent_dim), mean=0.)
            return z_mean + K.exp(z_log_var / 2) * epsilon

        z = Lambda(sampling, output_shape=(self.latent_dim,))([z_mean, z_log_var])

        decoder_h = self.generator

        decoder_mean = Dense(self.channels, activation='sigmoid')
        h_decoded = decoder_h(z)
        x_decoded_mean = decoder_mean(h_decoded)

        vae = Model(x, x_decoded_mean)

        print('VAE :')
        vae.summary()

        return vae

    def build_discriminator(self):

        model = Sequential()

        model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0, 1), (0, 1))))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))

        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    def train(self, epochs, batch_size=128, sample_interval=50):

        # Load the dataset
        (X_train, _), (_, _) = mnist.load_data()

        # Rescale -1 to 1
        X_train = X_train / 127.5 - 1.
        X_train = np.expand_dims(X_train, axis=3)

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random batch of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]

            print(imgs.shape)

            # Encoder and decoder images
            d_imgs = self.vae.predict(imgs)

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(d_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # Train VAE
            g_loss = self.vae.train_on_batch(imgs,d_imgs)

            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample_images(epoch,X_train)

    def sample_images(self, epoch,imgs):
        r, c = 5, 5

        idx = np.random.randint(0, imgs.shape[0], r*c)
        imgs = imgs[idx]

        gen_imgs = self.vae.predict(imgs)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,0],cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("images/%d.png" % epoch)
        plt.close()


if __name__ == '__main__':
    gan = GAN()
    gan.train(epochs=301, batch_size=32, sample_interval=30)