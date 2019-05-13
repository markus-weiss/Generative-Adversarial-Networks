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

# DISCRIMINATOR

def build_discriminator(img_shape):
    model = Sequential()
    model.add(Flatten(input_shape=img_shape))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(1)) # Ein Ausgabeneuron da nur eine klasse (gehört dazu oder nicht)
    model.add(Activation("sidmoid")) # warscheinlichkeit der zugehörigkeit
    model.summary()
    img = Input(shape=img_shape)
    d_pred = model(img)
    return Model(img, d_pred)

    