import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
import io
import matplotlib.pyplot as plt
import os



class ModelPartCheckpoint(tf.keras.callbacks.Callback):

    def __init__(self, checkpoint_dir, layers_to_save):
        super(ModelPartCheckpoint, self).__init__()
        self.filepath_templ = os.path.join(checkpoint_dir, "partial_save_{}_layer_{}({}).hdf5")

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.layers_to_save = layers_to_save


    def on_epoch_end(self, epoch, logs=None):

        for layer in self.layers_to_save:
            if isinstance(layer, str):
                l = self.model.get_layer(name=layer)
            elif isinstance(layer, int):
                l = self.model.get_layer(index=layer)
            else:
                raise ValueError("layers_to_save can only contain strings or integer layer indices!")

            filepath = self.filepath_templ.format(epoch, layer, l.name)
            l.save(filepath, overwrite=True, include_optimizer=False, save_format='h5')
