# Imports
import numpy as np
import matplotlib.pyplot as plt
from keras.models import *
from keras.layers import *
from keras.layers.normalization import *
from keras.optimizers import *
from model import *

import foolbox
from foolbox.attacks import SaliencyMapAttack 
from foolbox.attacks import LBFGSAttack
from foolbox.criteria import Misclassification
from foolbox.criteria import TargetClassProbability
from foolbox.criteria import TargetClass

# Load MNIST dataset
from data import *
mnist_data = MNIST()
x_train, y_train = mnist_data.get_train_set()
x_test, y_test = mnist_data.get_test_set()

# Softmax
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

# Load CNN
load = True
cnn = build_cnn()

if load == False:
    # TRAINING
    lr = 0.0005
    optimizer = Adam(lr=lr)
    cnn.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    cnn.fit(x_train, y_train, verbose=1,
            batch_size=128, nb_epoch=15,
            validation_data=(x_test, y_test))
    # TESTING
    score = cnn.evaluate(x_test, y_test)
    print("Test accuracy: ", score[1])
    # SAVE MODEL
    model_json = cnn.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    cnn.save_weights("model.h5")
    print("Saved model to disk")

if load == True:
    # LOAD MODEL
    json_file = open("model.json", "r")
    loaded_model_json = json_file.read()
    json_file.close()
    cnn = model_from_json(loaded_model_json)
    cnn.load_weights("model.h5")
    print("Loaded model from disk")

# FOOLING
fmodel = foolbox.models.KerasModel(cnn, bounds=(0, 1))

for _ in range(10):
    indx = np.random.randint(0, x_test.shape[0])
    image, label = x_test[indx], y_test[indx]

    print("Label: ", np.argmax(label))
    print("Prediction: ", np.argmax(fmodel.predictions(image)))
    # Apply attack
    attack = LBFGSAttack(fmodel, criterion=TargetClass(3))
    adversarial = attack(image, np.argmax(label))
    if adversarial is None: break
    print("Adversarial Prediction: ", np.argmax(fmodel.predictions(adversarial)))
    print("Adversarial Prediction: ", softmax(fmodel.predictions(adversarial)))

    # Plot the attack
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.title("Original")
    plt.imshow(image.reshape((28,28)), cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Adversarial")
    plt.imshow(adversarial.reshape((28,28)), cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Difference")
    difference = adversarial - image
    plt.imshow(difference.reshape((28,28)), cmap="gray")
    plt.axis("off")
    plt.show()