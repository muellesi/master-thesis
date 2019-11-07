import tensorflow as tf
from tensorflow import keras
from tensorflow_core.keras import datasets, layers, models, optimizers
from tensorflow_core.keras.preprocessing import image
from tensorflow_core.keras import applications
import numpy as np
import os



def make_model(input_shape=(256, 256, 1), output_shape=63):  # default output shape is 3*21 for 21 joints in a hand
    model = models.Sequential()
    model.add(layers.Conv2D(48, (5, 5), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(48 * 2, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(48 * 4, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(2048, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(output_shape, activation='relu'))
    model.summary()
    return model


def train_model(model, train_data, batch_size, max_epochs, validation_gen=None):
    model.compile(optimizer=keras.optimizers.Adam(0.01), loss=keras.losses.mean_squared_error, metrics=["mae", "acc"])

    if not os.path.exists("data/checkpoints/"):
        os.makedirs("data/checkpoints/")

    checkpointer = keras.callbacks.ModelCheckpoint(
            filepath=str("data/checkpoints/pose_est_weights.{epoch:02d}.hdf5"),
            save_best_only=False)
    tensorboard = keras.callbacks.TensorBoard(log_dir="data\\tensorboard_logs\\", histogram_freq=0,
                                              write_graph=True, write_images=True, update_freq='batch')
    prog = keras.callbacks.ProgbarLogger(count_mode='steps')
    model.fit(
            train_data,
            epochs=max_epochs,
            verbose=2,
            callbacks=[checkpointer, tensorboard, prog])
