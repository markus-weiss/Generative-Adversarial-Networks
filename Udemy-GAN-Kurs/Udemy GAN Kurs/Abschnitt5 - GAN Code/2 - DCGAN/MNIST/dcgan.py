# Immports
from keras.layers import *
from keras.layers.advanced_activations import *
from keras.layers.convolutional import *
from keras.models import *
from keras.optimizers import *
import matplotlib.pyplot as plt

import numpy as np
import sys

from g import *
from d import *
from mnist_data import *

PATH = "C:/Users/Jan/Dropbox/_Programmieren/Udemy GAN Kurs/Abschnitt5 - GAN Code/2 - DCGAN/"

# DCGAN Model Class
class DCGAN():
    def __init__(self):
        # Model parameters
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.z_dimension = 100
        optimizer = Adam(0.0002, 0.5)
        # BUILD DISCRIMINATOR
        self.discriminator = build_discriminator(self.img_shape)
        self.discriminator.compile(
            loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])
        # BUILD GENERATOR
        self.generator = build_generator(self.z_dimension, self.channels)
        # The generator takes noise as input and generates imgs
        z = Input(shape=(self.z_dimension,))
        img = self.generator(z)
        self.discriminator.trainable = False
        d_pred = self.discriminator(img)
        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, d_pred)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def train(self, epochs, batch_size=128, sample_interval=50):
        # Load and rescale dataset
        mnist_data = MNIST()
        x_train, _ = mnist_data.get_train_set()
        x_train = x_train / 127.5 - 1.
        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        # Start training
        for epoch in range(epochs):
            # TRAINSET IMAGES
            idx = np.random.randint(0, x_train.shape[0], batch_size)
            imgs = x_train[idx]
            # GENERATED IMAGES
            noise = np.random.normal(0, 1, (batch_size, self.z_dimension))
            gen_imgs = self.generator.predict(noise)
            # TRAIN DISCRIMINATOR
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            # TRAIN GENERATOR
            noise = np.random.normal(0, 1, (batch_size, self.z_dimension))
            g_loss = self.combined.train_on_batch(noise, valid)
            # SAVE PROGRESS
            if (epoch % sample_interval) == 0:
                print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % 
                    (epoch, d_loss[0], 100*d_loss[1], g_loss))
                self.sample_images(epoch)

    # Save sample images
    def sample_images(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.z_dimension))
        gen_imgs = self.generator.predict(noise)
        gen_imgs = 0.5 * gen_imgs + 0.5
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig(PATH + "images/MNIST/%d.png" % epoch)
        plt.close()

if __name__ == '__main__':
    dcgan = DCGAN()
    dcgan.train(epochs=200000, batch_size=32, sample_interval=1000)