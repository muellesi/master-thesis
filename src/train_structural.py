import tensorflow as tf
import os
import glob
import numpy as np
from sklearn.model_selection import train_test_split


def get_training_data(data_root):
    all_files = glob.glob(os.path.join(data_root, "**", "*.png"), recursive=True)
    all_files = np.random.shuffle(np.array(all_files))

    train_files =
    def data_generator(files_list):
        for file in files_list:
            yield file

    dataset = tf.data.Dataset.from_generator(data_generator())


def make_decoder():
    decoder = tf.keras.Sequential()
    decoder.add(tf.keras.layers.Conv2DTranspose(1280, 1, activation='relu'))
    decoder.add(tf.keras.layers.Conv2DTranspose(960, 1, activation='relu'))
    decoder.add(tf.keras.layers.Conv2DTranspose(576, 1, activation='relu'))
    decoder.add(tf.keras.layers.Conv2DTranspose(384, 1, activation='relu'))
    decoder.add(tf.keras.layers.Conv2DTranspose(192, 1, strides=2, activation='relu'))
    decoder.add(tf.keras.layers.Conv2DTranspose(144, 1, strides=2, activation='relu'))
    decoder.add(tf.keras.layers.Conv2DTranspose(144, 1, strides=2, activation='relu'))
    decoder.add(tf.keras.layers.Conv2DTranspose(96, 3, strides=2, activation='relu'))
    decoder.add(tf.keras.layers.Conv2DTranspose(32, 5, strides=2, activation='sigmoid'))
    return decoder


class AutoEncoder(tf.keras.Model):

    def __init__(self, name, img_dim, **kwargs):
        super(AutoEncoder, self).__init__(name=name, **kwargs)

        self.encoder = tf.keras.applications.MobileNetV2(include_top=False,
                                                         weights=None,
                                                         input_shape=img_dim,
                                                         **kwargs)

        self.decoder = make_decoder()


    def call(self, inputs, training=False):
        embedding = self.encoder(inputs)
        decoded = self.decoder(embedding)
        return tf.image.resize(decoded, self.input_shape)


def train_autoencoder(model):
    pass


if __name__ == '__main__':
    input_width = 224
    input_height = 224
    autoencoder = AutoEncoder(name="strct_autoenc", img_dim=(input_width, input_height, 1))

    dataset = get_training_data("E:\\MasterDaten\\Datasets\\StructuralLearning")
    train_autoencoder(autoencoder)
