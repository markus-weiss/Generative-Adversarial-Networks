# Imports
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import *
from keras.layers.normalization import *
from keras.optimizers import *
from keras.callbacks import EarlyStopping

# Load MNIST dataset
from data import *
mnist_data = MNIST()
x_train, y_train = mnist_data.get_train_set()
x_test, y_test = mnist_data.get_test_set()

# Define the CNN
model = Sequential()
# Conv Block 1
model.add(Conv2D(32, (5,5), input_shape=(28,28,1)))
model.add(Conv2D(64, (5,5)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Activation("relu"))
model.add(BatchNormalization())
# Conv Block 2
model.add(Conv2D(64, (3,3)))
model.add(Conv2D(128, (3,3)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Activation("relu"))
model.add(BatchNormalization())
# Fully connected layer 1
model.add(Flatten())
model.add(Dense(512))
model.add(Dropout(0.25))
model.add(Activation("relu"))
model.add(BatchNormalization())
# Output layer
model.add(Dense(10))
model.add(Activation("softmax"))

# Print the CNN layers
model.summary()

# Train the CNN
callbacks = [EarlyStopping(monitor="val_loss", patience=2)]
lr = 0.0005
optimizer = Adam(lr=lr)
model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
model.fit(x_train, y_train, verbose=1,
        batch_size=128, nb_epoch=100,
        validation_data=(x_test, y_test),
        callbacks=callbacks)

# Test the CNN
score = model.evaluate(x_test, y_test)
print("Test accuracy: ", score[1])