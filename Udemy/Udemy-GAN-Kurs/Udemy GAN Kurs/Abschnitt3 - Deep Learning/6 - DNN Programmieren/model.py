# Imports
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import *
from keras.layers.normalization import *
from keras.optimizers import *

# Load MNIST dataset
from data import *
mnist_data = MNIST()
x_train, y_train = mnist_data.get_train_set()
x_test, y_test = mnist_data.get_test_set()

# Define the DNN
model = Sequential()
# Hidden Layer 1
model.add(Dense(512, input_shape=(784,)))
model.add(Activation("relu"))
# Hidden Layer 2
model.add(Dense(512))
model.add(Activation("relu"))
# Output Layer
model.add(Dense(10))
model.add(Activation("softmax"))

# Print the DNN layers
model.summary()

# Train the DNN
lr = 0.0001
optimizer = Adam(lr=lr)
model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
model.fit(x_train, y_train, verbose=1,
        batch_size=128, nb_epoch=10,
        validation_data=(x_test, y_test))

# Test the DNN
score = model.evaluate(x_test, y_test)
print("Test accuracy: ", score[1])