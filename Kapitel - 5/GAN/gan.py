# Imports
## Keras
from keras.models import *
from keras.layers import *
from keras.layers.normalization import *
from keras.optimizers import *
from keras.layers.convolutional import *
from keras.layers.advanced_activations import *

## 
import numpy as np
import matplotlib.pyplot as plt

## load MNIST dataset
import data
from keras.callbacks import EarlyStopping



# Import Generator & Diskriminator & Data
from g import *
from d import *
from data import * 
PATH = "C:/Users/schnu/Desktop/UdemyML/UdemyGit/Kapitel - 5/GAN"

# GAN Model Class
class GAN():
    def __init__(self):
        # Model parameter
        self.img_rows = 28 # Pixelsize
        self.img_cols = 28 # Pixelsize
        self.channels = 1 # Farbkanäle
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        # Hyperparameter
        self.z_dimension = 100 # Zufallsvector
        learning_rate = 0.0002 # Learningrate
        beta_1 = 0.5 # Optimizer Value
        optimizer = Adam(lr=learning_rate, beta_1=beta_1) # Adam Optimizer

        # BUILD DISCRIMINATOR
        self.discriminator = build_discriminator(self.img_shape) # Bild erstellen und im Discriminator abspeichern
        self.discriminator.compile(
            loss="binary_crossentropy", # Wie schlimm ist, das ich etwas flasch gemacht habe # Fehlerfunktion
            optimizer=optimizer, # Optimizer hinzufügen 
            metrics=["accuracy"] # Metric hinzufügen
        )

        # BUILD GENERATOR
        self.generator = build_generator(self.z_dimension, self.img_shape)
        z = Input(shape=(self.z_dimension,))
        img = self.generator(z)
        self.discriminator.trainable = False
        d_pred = self.discriminator(img) # generatorBild übergeben an den Discriminator # An den gen wird noch einmal der Disk hinzugefügt also out von gen in disk
        self.combined = Model(z, d_pred) # Combination von Disk und Gen # wenn der Gen den Disk nicht besiegen kann müssen die Gewichte angepasst werden
        self.combined.compile(
            loss="binary_crossentropy",
            oprimizer=optimizer
        )

        
    
    def train(self, epochs, batch_size):
        # Load And rescale dataset # Neue Normoerung

        # Goal: [-1,1]
        # 0 = > -1
        # 127.5 => 0 
        # 255 => +1     255 / 127.5 = 2 -1 = 1

        mnist_data = MNIST()
        x_train, _ = mnist_data.get_train_set() # Kommt das Bild aus dem Training oder nicht / Wie viele Bilder ahben wir 
        c_train = x_train / 127.5 - 1.0 # Durch verwendung der tanh funktion muss rescaled werden (Siehe rechnung oben)

        # Adversarial ground truths # Bei Training werden kls 1 bilder (realbilder) übergeben
        valid = np.ones((batch_size, 1)) # Vector gefüllt mit 1 Numpy Array # RealBilder 1 = "echt" batch size viele
        fake = np.zeros((batch_size, 1)) # Vector gefüllt mit 0 Numpyx Array # FakeBilder

        # Start training 
        for epoch in range(epochs):
            # TrainSet Images
            # Batch_size = 4
            # 0, ..., 1499
            # 0 10 133 1337
            idx = np.random.randint(0, x_train.shape[0], batch_size) # Zufällige Zahl in Abhänigkeit # x_train.shape wie wiele Bilder, wie groß # batch_szie = wie viele bilder verwerdent werden sollen gleichzeitig
            imgs = x_train[idx] # Zufällige Bilder des Train set images 
            # Generate IMAGES
            noise = np.random.normal(0,1,(batch_size, self.z_dimension))
            gen_img = self.generator.predict(noise)

    # Wie gut ist das Modell gerade  
    def sample_images(self, epoch):
        pass


# Ist main datei, 
if __name__ == "__main__":
    gan = GAN() # Object erstellen 
    # gan.train(epochs=200000, batch_size=32, sample_interval=1000) # Ausführen 