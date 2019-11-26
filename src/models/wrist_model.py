import os

import tensorflow as tf
import tools



class WristCNN(tf.keras.Model):

    def __init__(self):
        super(WristCNN, self).__init__()
        self.encoder = tf.keras.applications.MobileNetV2(input_shape=(227, 227, 3), include_top=False, weights=None)
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(2048, activation=tf.keras.activations.relu, kernel_initializer=tf.keras.initializers.GlorotNormal(),
                                            bias_initializer='zeros')
        self.dense2 = tf.keras.layers.Dense(1024, activation=tf.keras.activations.relu, kernel_initializer=tf.keras.initializers.GlorotNormal(),
                                            bias_initializer='zeros')
        self.output_layer = tf.keras.layers.Dense(3, kernel_initializer=tf.keras.initializers.Orthogonal(gain=100.0), bias_initializer='zeros')


    def call(self, inputs, training=False):
        x = self.encoder(inputs)
        x = self.flatten(x)
        x = self.dense1(x, training=training)
        x = self.dense2(x, training=training)
        return self.output_layer(x)
