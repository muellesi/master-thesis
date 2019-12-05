import tensorflow as tf
import os
import glob
import numpy as np
import pandas as pd
import tools
import math
from tools import training_callbacks
import matplotlib.pyplot as plt
import models.models as models


data_dir = "E:\\MasterDaten\\Results\\structural"
__tensorboard_dir = 'tensorboard'
__checkpoint_dir = 'checkpoints'
net_input_width = 224
net_input_height = 224
batch_size = 20
max_epochs = 2000
learning_rate = 0.0005


def get_filepaths(filepaths):
    for file in filepaths:
        yield file


def decode_img(raw):
    img = tf.image.decode_png(raw, channels=1, dtype=tf.dtypes.uint16)
    return img


def load_image(path):
    img = tf.io.read_file(path)
    img = decode_img(img)
    return img


def scale_image(img):
    img = tf.cast(tf.image.resize(img, tf.constant([net_input_height, net_input_width], dtype=tf.dtypes.int32)), dtype=tf.float32)
    img = img / tf.constant(2500.0, dtype=tf.float32)
    img = tf.clip_by_value(img, clip_value_min=0.0, clip_value_max=1.0)  # ignore stuff more than 2.5m away.
    return img


def duplicate_image(img):
    return img, img  # for autoencoder - x == y


def batch_shuffle_prefetch(ds):
    # ds = ds.repeat()
    ds = ds.shuffle(batch_size * 20)
    ds = ds.batch(batch_size=batch_size)
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
    return ds


def add_random_noise(img):
    noise = tf.random.normal(shape=tf.shape(img), mean=0.0, stddev=0.01, dtype=tf.dtypes.float32)
    return img + noise


def add_random_bg_image(img, bg_images):
    bg = next(iter(bg_images))
    # For this to work we need to remove every pixel that
    # is NOT zero in the 'hand image' from the background
    # before adding both images
    mask = tf.math.sign(img)  # mask hand with 1, bg with 0. Everything that is now 1 has to be removed from bg
    cond = tf.math.equal(mask, tf.ones(tf.shape(mask)))  # convert to bool array
    mask = tf.where(cond, tf.zeros(tf.shape(mask)), tf.ones(tf.shape(mask)))  # use bool array to 'invert' mask -> former 1s are now 0s and vice versa
    return bg * mask + img

def prepare(ds, add_noise, bg_images=None):
    ds = ds.map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.map(scale_image, num_parallel_calls=tf.data.experimental.AUTOTUNE).cache()
    ds = ds.map(duplicate_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if bg_images:
        ds = ds.map(lambda img1, img2: (add_random_bg_image(img1, bg_images), img2))
    if add_noise:
        ds = ds.map(lambda img1, img2: (add_random_noise(img1), img2), num_parallel_calls=tf.data.experimental.AUTOTUNE)

    return batch_shuffle_prefetch(ds)


def get_data(data_root, val_split=0.33):
    all_files = glob.glob(os.path.join(data_root, "hands", "**", "*.png"), recursive=True)
    all_files = np.array(all_files)
    np.random.shuffle(all_files)

    val_last_index = int(round(val_split * len(all_files)))
    files_val = all_files[:val_last_index]
    files_train = all_files[val_last_index:]

    augment_backgrounds = tf.data.Dataset.list_files(os.path.join(data_root, "augmentation", "**", "*.png"), shuffle=True)
    augment_backgrounds = augment_backgrounds.map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    augment_backgrounds = augment_backgrounds.map(scale_image, num_parallel_calls=tf.data.experimental.AUTOTUNE).cache()

    print("Iterating backgrounds once to cache all images...")
    for img in augment_backgrounds:  # cache all images...
        pass
    print("done!")
    augment_backgrounds = augment_backgrounds.repeat().shuffle(5).prefetch(tf.data.experimental.AUTOTUNE)

    ds_train = tf.data.Dataset.from_tensor_slices(np.array(files_train)).cache()
    ds_train = prepare(ds_train, add_noise=True, bg_images=augment_backgrounds)

    ds_val = tf.data.Dataset.from_tensor_slices(np.array(files_val)).cache()
    ds_val = prepare(ds_val, add_noise=True, bg_images=augment_backgrounds)

    return ds_train, ds_val


def get_autoencoder(img_dim):
    encoder = models.make_model('imagenetv2', input_shape=img_dim)
    decoder = models.make_model('ae-head')
    ae = tf.keras.Sequential()
    ae.add(encoder)
    ae.add(decoder)
    return ae

def train_autoencoder(model, train_dataset, validation_dataset, learning_rate=0.01):
    tensorboard_dir = os.path.join(data_dir, __tensorboard_dir)
    checkpoint_dir = os.path.join(data_dir, __checkpoint_dir)

    tools.clean_tensorboard_logs(tensorboard_dir)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.binary_crossentropy,
                  metrics=["mae", "acc", "mse"])

    if not os.path.exists(checkpoint_dir): os.makedirs(checkpoint_dir)
    if not os.path.exists(tensorboard_dir): os.makedirs(tensorboard_dir)

    checkpointer = tf.keras.callbacks.ModelCheckpoint(
            monitor='val_loss',
            filepath=os.path.join(checkpoint_dir, "full_ae_weights.epoch_{epoch:02d}.hdf5"),
            save_best_only=True)

    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_dir,
                                                 histogram_freq=1,
                                                 write_graph=True,
                                                 write_images=True,
                                                 update_freq='batch',
                                                 profile_batch=0)

    visu = tools.training_callbacks.AEVisuCallback(
            log_dir=tensorboard_dir + "\\plots\\",
            feed_inputs_display=validation_dataset,
            plot_every_x_batches=200
            )

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.75,
                                                     patience=7, min_lr=0.00000001, verbose=1)

    progbar = tf.keras.callbacks.ProgbarLogger(count_mode='steps')

    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=45, verbose=1)

    encoder_saver = tools.training_callbacks.ModelPartCheckpoint(os.path.join(checkpoint_dir, "partial"), [0, 1])

    model.fit(
            train_dataset,
            # steps_per_epoch=5000,
            validation_data=validation_dataset,
            # validation_steps=300,
            epochs=max_epochs,
            verbose=0,
            callbacks=[checkpointer, tensorboard, reduce_lr, visu, early_stop, encoder_saver])


if __name__ == '__main__':
    ds_train, ds_val = get_data("E:\\MasterDaten\\Datasets\\StructuralLearning")
    # batch = ds_train.take(1)
    #
    # data = batch.unbatch()
    # for inp, outp in data:
    #     fig = plt.figure()
    #     input = np.squeeze(inp)
    #     output = np.squeeze(outp)
    #     ax1 = fig.add_subplot(221)
    #     ax1.imshow(input)
    #     ax2 = fig.add_subplot(222)
    #     ax2.imshow(output)
    #     fig.show()

    autoencoder = get_autoencoder(img_dim=(net_input_width, net_input_height, 1))
    autoencoder.summary()

    train_autoencoder(autoencoder, ds_train, ds_val, learning_rate=learning_rate)
