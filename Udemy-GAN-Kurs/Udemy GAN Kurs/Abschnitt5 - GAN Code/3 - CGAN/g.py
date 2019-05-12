# Immports
from keras.layers import *
from keras.layers.advanced_activations import *
from keras.layers.convolutional import *
from keras.models import *
from keras.optimizers import *
import matplotlib.pyplot as plt
import numpy as np
 
def build_generator(z_dimension, num_classes, img_shape):
    noise = Input(shape=(z_dimension,))
    label = Input(shape=(num_classes,))

    x = Concatenate()([noise, label])

    x = Dense(512)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = Dense(1024)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = Dense(np.prod(img_shape))(x)
    x = Activation("tanh")(x)
    img = Reshape(img_shape)(x)

    model = Model(input=[noise, label], output=img)
    model.summary()
    return model