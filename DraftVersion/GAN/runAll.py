import bigan
import dcgan
import wgan
import gan

EPO = 20000
BAT_SIZE = 64
INTERVAL = 19999



if __name__ == '__main__':

    dcgan = dcgan.DCGAN()
    #dcgan.train(epochs=EPO, batch_size=BAT_SIZE, save_interval=INTERVAL)
    del dcgan

    bigan = bigan.BIGAN()
    #bigan.train(epochs=EPO, batch_size=BAT_SIZE, sample_interval=INTERVAL)
    del bigan

    wgan = wgan.WGAN()
    #wgan.train(epochs=EPO, batch_size=BAT_SIZE, sample_interval=INTERVAL)
    del wgan

    gan = gan.GAN()
    del gan
