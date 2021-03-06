# Immports
from keras.layers import *
from keras.layers.advanced_activations import *
from keras.layers.convolutional import *
from keras.models import *
from keras.optimizers import *
import matplotlib.pyplot as plt

def build_discriminator(img_shape):
    model = Sequential() #28x28
    
    model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=img_shape, padding="same")) #14x14
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, kernel_size=3, strides=2, padding="same")) #7x7
    model.add(ZeroPadding2D(padding=((0,1),(0,1)))) # 8x8
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, kernel_size=3, strides=2, padding="same")) #4x4
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))

    model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))

    model.add(Flatten()) #4x4x256 => 16 x 256 = 2^4 x 2^8 = 2^12 = 4096 
    model.add(Dense(1))
    model.add(Activation("sigmoid"))

    model.summary()
    img = Input(shape=img_shape)
    d_pred = model(img)
    return Model(input=img, output=d_pred)