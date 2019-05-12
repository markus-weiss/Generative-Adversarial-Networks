# Imports
from keras.models import Sequential
from keras.layers import *
from keras.layers.normalization import *
from keras.optimizers import *
from keras.layers.convolutional import *
from keras.layers.advanced_activations import *

import numpy as np
import matplotlib.pyplot as plt

# Load MNIST dataset
from g import *
from d import *
from data import *
PATH = "C:/Users/Jan/Dropbox/_Programmieren/Udemy GAN Kurs/Abschnitt5 - GAN Code/1 - GAN/"

# GAN Model Class
class GAN():
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
        self.generator = build_generator(self.z_dimension, self.img_shape)
        z = Input(shape=(self.z_dimension,))
        img = self.generator(z)
        self.discriminator.trainable = False
        d_pred = self.discriminator(img)
        self.combined = Model(z, d_pred)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def train(self, epochs, batch_size, sample_interval):
        # Load and rescale dataset
        mnist_data = MNIST()
        x_train, _ = mnist_data.get_train_set()
        x_train = x_train / 127.5 - 1.0
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
                print("[D loss: ", d_loss[0], "acc: ", round(d_loss[1]*100, 2), "] [G loss: ", g_loss, "]")
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
        fig.savefig(PATH + "images/%d.png" % epoch)
        plt.close()


if __name__ == '__main__':
    gan = GAN()
    gan.train(epochs=200000, batch_size=32, sample_interval=1000)