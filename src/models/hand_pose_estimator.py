import tensorflow as tf
import numpy

from tensorflow.keras import datasets, layers, models


def make_model(input_shape=(32, 32, 32, 3), output_shape=63): # default output shape is 3*21 for 21 joints in a hand
    model = models.Sequential()
    model.add(layers.Conv3D(96, (5, 5, 5), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling3D((2, 2, 2)))
    model.add(layers.Conv3D(192, (3, 3, 3), activation='relu'))
    model.add(layers.MaxPooling3D((2, 2, 2)))
    model.add(layers.Conv3D(384, (3, 3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(4096, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(output_shape, activation='relu'))
    model.summary()
    return model

def train_model(model):
    pass
