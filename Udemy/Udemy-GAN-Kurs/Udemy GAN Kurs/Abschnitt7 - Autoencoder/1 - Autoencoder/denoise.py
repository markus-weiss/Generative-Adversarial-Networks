# Imports
from keras.layers import *
from keras.layers.advanced_activations import *
from keras.layers.convolutional import *
from keras.models import *
from keras.optimizers import *
import matplotlib.pyplot as plt
import numpy as np
from mnist_data import *

# Load MNIST dataset
data = MNIST()
x_train, _ = data.get_train_set()
x_test, _ = data.get_test_set()

x_train_noise = x_train + 0.2 * np.random.normal(size=(x_train.shape))
x_test_noise = x_test + 0.2* np.random.normal(size=(x_test.shape))

#plt.imshow(x_test_noise[0].reshape(28,28), cmap="gray")
#plt.show()

# Encoded dimension
encoding_dim = 98

# Keras Model: Autoencoder
# Input Tensors
input_img = Input(shape=(28,28,1,))
# Encoder Part
x = Conv2D(8, kernel_size=3, activation="relu", padding="same")(input_img) # 28x28x8
x = MaxPooling2D(padding="same")(x) # 14x14x8
x = Conv2D(4, kernel_size=3, activation="relu", padding="same")(x) # 14x14x4
x = MaxPooling2D(padding="same")(x) # 7x7x4
encoded = Conv2D(2, kernel_size=3, activation="relu", padding="same")(x) # 7x7x2
# Decoder Part
x = Conv2D(4, kernel_size=3, activation="relu", padding="same")(encoded) # 7x7x4
x = UpSampling2D()(x) # 14x14x4
x = Conv2D(8, kernel_size=3, activation="relu", padding="same")(x) # 14x14x8
x = UpSampling2D()(x) # 28x28x8
decoded = Conv2D(1, kernel_size=3, activation="sigmoid", padding="same")(x) # 28x28x1
# Output Tensors
autoencoder = Model(input=input_img, output=decoded)

# Training
autoencoder.compile(optimizer="adadelta", loss="binary_crossentropy")
autoencoder.fit(x_train_noise, x_train, epochs=5, batch_size=128, validation_data=(x_test_noise, x_test))

# Testing
test_images = x_test_noise[:10]
decoded_imgs = autoencoder.predict(test_images)

# PLot test images
plt.figure(figsize=(12,6))
for i in range(10):
    # Original image
    ax = plt.subplot(2 , 10, i+1)
    plt.imshow(test_images[i].reshape(28,28), cmap="gray")
    # Decoded image
    ax = plt.subplot(2 , 10, i+1+10)
    plt.imshow(decoded_imgs[i].reshape(28,28), cmap="gray")
plt.show()