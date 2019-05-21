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

# Define DNN
model = Sequential()
# Hidden Layer 1
model.add(Dense(256, input_shape=(784,))) # 512 * 784 + 512
model.add(Activation("relu"))
# Hidden Layer 2
model.add(Dense(256))
model.add(Activation("relu"))
# Hidden Layer 2
# Outputlayer
model.add(Dense(10))
model.add(Activation("softmax"))

# Print the DNN layers
model.summary()

# Train the DNN
learningRate = 0.001
optimizer = Adam(lr=learningRate)
model.compile(loss="categorical_crossentropy", optimizer= optimizer, metrics=["accuracy"]) # Fehlerfkt , optimizer , metric= genauigkeit
model.fit(x_train, y_train, # Train, übergabe 
    verbose=1, # verbose = consolOutput
    batch_size=64, # Wie viele Bilder Parallel
    nb_epoch=20, # Anzahl der Epochen
    validation_data=(x_test, y_test) # Prüfdaten
)

# Test the DNN
score = model.evaluate(x_test, y_test)
print("Test Accuracy:" , score[1])
# 0,9778
# 0.9814 - 0.001 E20
# 0.9824 - 0.001 E20 batch 64
# 0.9819 - 0.001 E20 batch 64 Dense 256 256
# 0.9816 - 0.001 E20 batch 64 Dense 784 350 100 
