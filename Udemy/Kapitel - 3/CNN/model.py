# Imports
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import *
from keras.layers.normalization import *
from keras.optimizers import *
import data
from keras.callbacks import EarlyStopping

# Load MNIST dataset
from data import *
mnist_data = MNIST()
x_train, y_train = mnist_data.get_train_set()
x_test, y_test = mnist_data.get_test_set()

# Define DNN
model = Sequential()

# Conv Block mit InputLayer
model.add(Conv2D(32, (7,7), input_shape=(28,28,1))) # 32(zufällig) Filter jeder ist 7*7 groß, input = bildgröße
model.add(Conv2D(32, (7,7)))
model.add(MaxPooling2D(pool_size=(2,2))) # Reduktion der Bildgröße aber wichtige werte werden behalten!
model.add(Activation("relu")) # Activierungsfuntion  
model.add(BatchNormalization()) # Daten werden um Mittelwert und Varianz normalisiert / in abhänigkeit der Gewichtung

# Conv Block 2
model.add(Conv2D(64, (5,5))) # 32(zufällig) Filter jeder ist 7*7 groß, 
model.add(Conv2D(128, (3,3)))
model.add(MaxPooling2D(pool_size=(2,2))) # Reduktion der Bildgröße
model.add(Activation("relu")) # Activierungsfuntion
model.add(BatchNormalization())

 # Fully Connected Layer muss immer am Ende sein  
model.add(Flatten()) # Formit bilder zur Vektor Anordnung
model.add(Dense(512)) # Dense Layer 
model.add(Dropout(0.25)) # Neuronen die möglicherweise zu stark gewichtet sein könnten werden ausgeschalten
model.add(Activation("relu"))
model.add(BatchNormalization())

# Output layer
model.add(Dense(10))
model.add(Activation("softmax"))


# Print the DNN layers
model.summary()

# Train the DNN
callbacks = [EarlyStopping(monitor="acc", patience=2)] # Welcher Parameter in Monitor, und wie viel Epochen er schlechter werden darf
learningRate = 0.001
optimizer = Adam(lr=learningRate)
model.compile(loss="categorical_crossentropy", optimizer= optimizer, metrics=["accuracy"]) # Fehlerfkt , optimizer , metric= genauigkeit
model.fit(x_train, y_train, # Train, übergabe 
    verbose=1, # verbose = consolOutput
    batch_size=64, # Wie viele Bilder Parallel
    epochs=100, # Anzahl der Epochen
    validation_data=(x_test, y_test), # Prüfdaten
    callbacks=callbacks # Übergibt die Callbackbedingungen
)

# Test the DNN
score = model.evaluate(x_test, y_test)
print("Test Accuracy:" , score[1])
