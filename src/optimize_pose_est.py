import tensorflow as tf
import os
import glob
import numpy as np
import pandas as pd
import tools
import math
from tools import training_callbacks
import matplotlib.pyplot as plt
from datasets import SerializedDataset
import json
from tensorboard.plugins.hparams import api as hp
from datasets.tfrecord_helper import depth_and_skel
import copy



net_input_width = 224
net_input_height = 224
batch_size = 50


def scale_image(img, skel):
    img = tf.cast(tf.image.resize(img, tf.constant([net_input_height, net_input_width], dtype=tf.dtypes.int32)), dtype=tf.float32)
    img = img / tf.constant(2500.0, dtype=tf.float32)
    img = tf.clip_by_value(img, clip_value_min=0.0, clip_value_max=1.0)  # ignore stuff more than 2.5m away.
    return img, skel


def batch_shuffle_prefetch(ds):
    ds = ds.shuffle(batch_size * 20)
    ds = ds.batch(batch_size=batch_size)
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
    return ds


def add_random_noise(img, skel):
    noise = tf.random.normal(shape=tf.shape(img), mean=0.0, stddev=0.01, dtype=tf.dtypes.float32)
    return img + noise, skel


def prepare_ds(ds, add_noise):
    ds = ds.map(scale_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if add_noise:
        ds = ds.map(add_random_noise, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    return batch_shuffle_prefetch(ds)


def train_model(model, train_dataset, validation_dataset, test_dataset, max_epochs, learning_rate):
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.mean_squared_error,
                  metrics=['accuracy'])

    early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3, verbose=1)
    term_on_nan = tf.keras.callbacks.TerminateOnNaN()
    # reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=2, min_lr=0.00000001)

    model.fit(
            x=train_dataset,
            epochs=max_epochs,
            verbose=1,
            callbacks=[early_stop, term_on_nan])
    _, accuracy = model.evaluate(test_dataset)
    return accuracy


def make_model(encoder_pretrained, encoder_trainable, dropout, bottleneck_size, activation):
    encoder = tf.keras.models.load_model(encoder_pretrained)
    encoder.trainable = encoder_trainable

    regressor = tf.keras.models.Sequential()
    regressor.add(tf.keras.layers.Flatten(input_shape=encoder.output_shape[1:]))

    # FC 1
    regressor.add(tf.keras.layers.Dense(2048,
                                        kernel_initializer=tf.keras.initializers.GlorotNormal(),
                                        bias_initializer='zeros'))
    regressor.add(tf.keras.layers.Activation(activation))
    if dropout: regressor.add(tf.keras.layers.Dropout(dropout))

    # FC 2
    regressor.add(tf.keras.layers.Dense(1024,
                                        kernel_initializer=tf.keras.initializers.GlorotNormal(),
                                        bias_initializer='zeros'))
    regressor.add(tf.keras.layers.Activation(activation))
    if dropout: regressor.add(tf.keras.layers.Dropout(dropout))

    # FC 3 - Bottleneck
    if bottleneck_size:
        regressor.add(tf.keras.layers.Dense(bottleneck_size))
        regressor.add(tf.keras.layers.Activation(activation))

    regressor.add(tf.keras.layers.Dense(63,
                                        kernel_initializer=tf.keras.initializers.Orthogonal(gain=100.0),
                                        bias_initializer='zeros'))
    regressor.summary()

    model = tf.keras.models.Sequential()
    model.add(encoder)
    model.add(regressor)
    model.summary()
    return model


def run_experiment(run_dir, hparams, ds_train, ds_val, ds_test):
    with tf.summary.create_file_writer(run_dir).as_default():
        hp.hparams(hparams)  # record the values used in this trial
        dropout = hparams[HP_DROPOUT] / 10.0 if hparams[HP_DROPOUT] > 0 else None
        bottleneck = hparams[HP_BOTTLENECK_SIZE] if hparams[HP_BOTTLENECK_SIZE] > 0 else None
        model = make_model(encoder_pretrained="data\\structural\\ae_with_bg_high_acc\\partial_save_46_layer_0(mobilenetv2_1.00_224).hdf5",
                           encoder_trainable=hparams[HP_ENCODER_TRAINABLE],
                           dropout=dropout,
                           bottleneck_size=bottleneck,
                           activation=hparams[HP_ACTIVATION]
                           )
        accuracy = train_model(
                model=model,
                train_dataset=ds_train,
                validation_dataset=ds_val,
                test_dataset=ds_test,
                learning_rate=hparams[HP_LEARN_RATE] * 10 ** (-5),
                max_epochs=30,
                )
        tf.summary.scalar(METRIC_ACCURACY, accuracy, step=1)
        del model


def grid_search():
    with open("datasets.json", "r") as f:
        ds_settings = json.load(f)

    ds_provider = SerializedDataset(ds_settings["BigHands"])

    ds_train = ds_provider.get_data("train")
    ds_train = ds_train.map(depth_and_skel, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_train = prepare_ds(ds_train, True)

    ds_val = ds_provider.get_data("validation")
    ds_val = ds_val.map(depth_and_skel, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_val = prepare_ds(ds_val, True)

    ds_test = ds_provider.get_data("test")
    ds_test = ds_test.map(depth_and_skel, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_test = prepare_ds(ds_test, False)

    session_num = 0

    runs = []
    for learn_rate in HP_LEARN_RATE.domain.values:
        for dropout_rate in HP_DROPOUT.domain.values:
            for bottleneck_size in HP_BOTTLENECK_SIZE.domain.values:
                for activation_function in HP_ACTIVATION.domain.values:
                    for encoder_trainable in HP_ENCODER_TRAINABLE.domain.values:
                        hparams = {
                                HP_LEARN_RATE: learn_rate,
                                HP_DROPOUT: dropout_rate,
                                HP_BOTTLENECK_SIZE: bottleneck_size,
                                HP_ACTIVATION: activation_function,
                                HP_ENCODER_TRAINABLE: encoder_trainable
                                }
                        runs.append(hparams)

    for run in runs:
        run_name = "run-{}".format(session_num)
        if not os.path.exists(os.path.join(log_dir, run_name)):
            print('--- Starting trial: {}'.format(run_name))
            print({h.name: run[h] for h in run})
            run_experiment(run_dir=os.path.join(log_dir, run_name), hparams=run,
                           ds_train=ds_train, ds_val=ds_val, ds_test=ds_test)
        else:
            print("Experiment {} was already run/is running elsewhere! Not running again!".format(run_name))
        session_num += 1


if __name__ == '__main__':
    log_dir = "E:\\MasterDaten\\Results\\optimization\\pose_est_tuning"

    HP_LEARN_RATE = hp.HParam('learning_rate',
                              display_name='Learning rate [x10^5]',
                              description='Learning rate [x10<sup>5</sup>]',
                              domain=hp.Discrete([1000, 500, 100, 50, 10, 5, 1]))
    HP_DROPOUT = hp.HParam('dropout',
                           display_name='Dropout Rate [x10]',
                           description='[x10]',
                           domain=hp.Discrete([-1, 2, 4]))
    HP_BOTTLENECK_SIZE = hp.HParam('bottleneck_size',
                                   display_name='Bottleneck Size',
                                   description='Bottleneck Size',
                                   domain=hp.Discrete([-1, 30, 35, 40, 45]))
    HP_ACTIVATION = hp.HParam('activation',
                              display_name='Activation Function',
                              description='Activation Function',
                              domain=hp.Discrete(["relu", "tanh"]))
    HP_ENCODER_TRAINABLE = hp.HParam('encoder_trainable',
                                     display_name='Encoder Trainable',
                                     description='Encoder Trainable',
                                     domain=hp.Discrete([True, False]))

    METRIC_ACCURACY = 'accuracy'

    with tf.summary.create_file_writer(log_dir).as_default():
        hp.hparams_config(
                hparams=[HP_LEARN_RATE, HP_DROPOUT, HP_BOTTLENECK_SIZE, HP_ACTIVATION, HP_ENCODER_TRAINABLE],
                metrics=[hp.Metric(METRIC_ACCURACY, display_name='Accuracy')],
                )

    grid_search()
