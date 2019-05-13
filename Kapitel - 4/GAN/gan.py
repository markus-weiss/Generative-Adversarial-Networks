# Imports
## Keras
from keras.models import Sequential
from keras.layers import *
from keras.layers.normalization import *
from keras.optimizers import *
from keras.layers.convolutional import *
from keras.layers.advanced_activations import *

## 
import numpy as np
import matplotlib.pyplot as plt

## load MNIST dataset
import data
from keras.callbacks import EarlyStopping

# Import Generator & Diskriminator & Data
from g import *
from d import *
from data import *
PATH = "C:\Users\schnu\Desktop\UdemyML\UdemyGit\Kapitel - 5\GAN"
    