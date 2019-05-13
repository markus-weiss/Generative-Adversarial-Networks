# Imports
## Keras
from keras.models import *
from keras.layers import *
from keras.layers.normalization import *
from keras.optimizers import *
from keras.layers.convolutional import *
from keras.layers.advanced_activations import *


# GENERATOR

def build_generator(z_dimension, img_shape):
    model = Sequential() # Model anlegen
    # InputLayer
    model.add(Dense(256, input_dim=z_dimension)) # Denselayer und Input Dimension
    model.add(LeakyReLU(alpha=0.2)) # Aktivierungsfunktion Wie RELU nur nicht auf null sonder winkel bei 0.2pi
    model.add(BatchNormalization(momentum=0.8)) # NetzwerkOptimierung
    # Layer 1
    model.add(Dense(512)) # Denselayer und Input Dimension
    model.add(LeakyReLU(alpha=0.2)) # Aktivierungsfunktion Wie RELU nur nicht auf null sonder winkel bei 0.2pi
    model.add(BatchNormalization(momentum=0.8)) # NetzwerkOptimierung  1
    # Layer 2
    model.add(Dense(1024)) # Denselayer und Input Dimension
    model.add(LeakyReLU(alpha=0.2)) # Aktivierungsfunktion Wie RELU nur nicht auf null sonder winkel bei 0.2pi
    model.add(BatchNormalization(momentum=0.8)) # NetzwerkOptimierung
    # Outputlänge des Vectors 28 * 28 = 784
    model.add(Dense(np.prod(img_shape))) # Errechtent die Output Länge
    model.add(Activation("tanh")) # Möglichkeit zur verbesserung des Trainings
    model.add(Reshape(img_shape)) # Erzeugt das Bild und in img gespeichert
    # Ausgabe des Models
    model.summary()
    noise = Input(shape=(z_dimension,)) # Noise Tensor um Vector zu speichern
    img = model(noise)
    return Model(noise, img)
