from __future__ import print_function, division

from keras import metrics
from keras.datasets import mnist
from keras import backend as K
from keras.layers import Input, Dense, Reshape, Flatten, Lambda,Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D,BatchNormalization,Reshape
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D,Conv2DTranspose
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.utils import to_categorical

from scipy.stats import norm

import matplotlib.pyplot as plt

import sys

import numpy as np

class VAE():
    def __init__(self):
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 256

        optimizer = Adam(0.0005, 0.5)

        # Build the encoder
        self.encoder = self.build_encoder()

        # Build the generator
        self.generator = self.build_generator()

        # Build the Variational Auto-encoder
        self.vae = self.build_vae()
        self.vae.compile(loss='binary_crossentropy', optimizer=optimizer)


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

        x = Input(shape=self.img_shape,name='def')

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
        xent_loss = self.img_rows * self.img_cols * metrics.binary_crossentropy(x, x_decoded_mean)
        kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        vae_loss = K.mean(xent_loss + kl_loss)

        vae.add_loss(vae_loss)
        vae.compile(optimizer='rmsprop')
        vae.summary()

        return vae

    def train(self, epochs, batch_size=32, sample_interval=50):
        # Load the dataset
        (x_train, y_train), (x_test, y_test_) = mnist.load_data()
        x_train = x_train.astype('float32') / 255.
        x_test = x_test.astype('float32') / 255.
        print(np.array(x_train).shape)
        x_train = np.array(np.expand_dims(x_train,axis=3))
        x_test = np.array(np.expand_dims(x_test,axis=3))
        print(np.array(x_train).shape)
        # x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
        # x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
        # print(np.array(x_train).shape)
        y_train = to_categorical(y_train, 10)
        y_test = to_categorical(y_test_, 10)

        self.vae.fit(x_train,
                shuffle=True,
                epochs=epochs)

        n = 15  # figure with 15x15 digits
        digit_size = 28
        figure = np.zeros((digit_size * n, digit_size * n))


        grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
        grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

        for i, yi in enumerate(grid_x):
            for j, xi in enumerate(grid_y):
                z_sample = np.array([[xi, yi]])
                x_decoded = self.generator.predict(z_sample)
                digit = x_decoded[0].reshape(digit_size, digit_size)
                figure[i * digit_size: (i + 1) * digit_size,
                j * digit_size: (j + 1) * digit_size] = digit

        plt.figure(figsize=(10, 10))
        plt.imshow(figure, cmap='Greys_r')
        plt.show()

if __name__ == '__main__':
    vae = VAE()
    vae.train(epochs=301, batch_size=32, sample_interval=30)