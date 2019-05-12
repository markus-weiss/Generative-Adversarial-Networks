# Immports
from keras.layers import *
from keras.layers.advanced_activations import *
from keras.layers.convolutional import *
from keras.models import *
from keras.optimizers import *
import matplotlib.pyplot as plt

def build_generator(z_dimension, channels):
    model = Sequential()
    
    model.add(Dense(128 * 7 * 7, input_dim=z_dimension))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Reshape((7, 7, 128)))

    model.add(UpSampling2D())
    model.add(Conv2D(128, kernel_size=3, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))

    model.add(UpSampling2D())
    model.add(Conv2D(64, kernel_size=3, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2D(channels, kernel_size=3, padding="same"))
    model.add(Activation("tanh"))

    model.summary()
    noise = Input(shape=(z_dimension,))
    img = model(noise)
    return Model(input=noise, output=img)