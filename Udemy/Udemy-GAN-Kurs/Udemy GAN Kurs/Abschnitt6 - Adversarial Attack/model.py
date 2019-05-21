# Imports
import numpy as np
import matplotlib.pyplot as plt
from keras.models import *
from keras.layers import *
from keras.layers.normalization import *
from keras.optimizers import *

# Load MNIST dataset
from data import *
mnist_data = MNIST()
x_train, y_train = mnist_data.get_train_set()
x_test, y_test = mnist_data.get_test_set()

def build_cnn():
        # Define the CNN
        model = Sequential()
        # Conv Block 1
        model.add(Conv2D(32, (7,7), input_shape=(28,28,1)))
        model.add(Conv2D(32, (5,5)))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Activation("relu"))
        # Conv Block 2
        model.add(Conv2D(64, (5,5)))
        model.add(Conv2D(128, (3,3)))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Activation("relu"))
        # Fully connected layer 1
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation("relu"))
        # Fully connected layer 1
        model.add(Dense(256))
        model.add(Activation("relu"))
        # Output layer
        model.add(Dense(10))
        model.add(Activation("softmax"))
        # Print the CNN layers
        model.summary()
        # Model object
        img = Input(shape=(28,28,1))
        pred = model(img)
        return Model(input=img, output=pred)