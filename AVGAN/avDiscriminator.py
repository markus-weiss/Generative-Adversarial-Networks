import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image


im1 = cv2.imread('img1.jpg')
im2 = cv2.imread('img2.jpg')

merged = np.concatenate((im1, im2), axis=2) # creates a numpy array with 6 channels

arr = np.ones((280, 280, 6), dtype=np.uint8) * 255
img=Image.fromarray(arr)
img.show()


for i in range(280) :
   for j in range(280):
       arr[i,j,0]=im1[i,j,0]
       arr[i,j,1]=im1[i,j,1]
       arr[i,j,2]=im1[i,j,2]
       arr[i,j,3]=im2[i,j,0]
       arr[i,j,4]=im2[i,j,1]
       arr[i,j,5]=im2[i,j,2]
img=Image.fromarray(arr)
img.show()


# def discriminator_Image():
#     model = tf.keras.Sequential()
#     model.add(layers.Conv2D(32, (3,3), stride = (1,1), padding = 'same', input_shape= [280,280,3]))
#     model.add(layers.LeakyReLU())

#     model.add(MaxPooling2D(2, 2))
#     model.add(layers.Conv2D(64, (3,3), stride = (1,1), padding = 'same'))
#     model.add(layers.LeakyReLU())

#     model.add(MaxPooling2D(2, 2))
#     model.add(layers.Conv2D(128, (3,3), stride = (1,1), padding = 'same'))
#     model.add(layers.LeakyReLU())
#     model.add(MaxPooling2D(2, 2))
  
#     return model




# def discriminator_Audio():
   
#     model = tf.keras.Sequential()
#     model.add(layers.Conv2D(32, (3,3), stride = (1,1), padding = 'same', input_shape= [280,280,2]))
#     model.add(layers.LeakyReLU())

#     model.add(MaxPooling2D(2, 2))
#     model.add(layers.Conv2D(64, (3,3), stride = (1,1), padding = 'same'))
#     model.add(layers.LeakyReLU())

#     model.add(MaxPooling2D(2, 2))
#     model.add(layers.Conv2D(128, (3,3), stride = (1,1), padding = 'same'))
#     model.add(layers.LeakyReLU())
#     model.add(MaxPooling2D(2, 2))
    





#     # model.add(layers.Flatten())
#     # model.add(layers.Dense(1))
  
#     return model