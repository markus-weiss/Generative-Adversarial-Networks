# Immports
from keras.layers import *
from keras.layers.advanced_activations import *
from keras.layers.convolutional import *
from keras.models import *
from keras.optimizers import *
import matplotlib.pyplot as plt
 
def build_discriminator(img_shape, num_classes):
    img = Input(shape=img_shape)
    label = Input(shape=(num_classes,))

    img_flatten = Flatten()(img)
    x = Concatenate()([img_flatten, label])

    x = Dense(512)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.25)(x)
    x = Dense(512)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.25)(x)
    x = Dense(1)(x)
    d_pred = Activation("sigmoid")(x)

    model = Model(input=[img, label], output=d_pred)
    model.summary()
    return model