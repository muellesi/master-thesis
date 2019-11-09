import tensorflow as tf
from tensorflow import keras
from tensorflow_core.keras import datasets, layers, models, optimizers
from tensorflow_core.keras.preprocessing import image
from tensorflow_core.keras import applications
import numpy as np
import tools
import os



__checkpoint_dir = "data\\pose_est\\checkpoints\\"
__tensorboard_dir = "data\\pose_est\\tensorboard_logs\\"
__checkpoint_file_prefix = "pose_est_"
__logger = tools.get_logger(__name__, do_file_logging=False)


def __try_load_checkpoint(model, checkpoint_dir: str):
    """
    Tries to load a checkpoint file for the specified model
    :param checkpoint_dir: directory where the checkpoints are saved
    :return: Model with loaded checkpoint if checkpoint existed, else unmodified model
    """
    if os.path.exists(checkpoint_dir):
        cp_files = [os.path.abspath(os.path.join(checkpoint_dir, filename)) for filename in os.listdir(checkpoint_dir)]
        cp_files = [path for path in cp_files if os.path.isfile(path) and __checkpoint_file_prefix in os.path.basename(path)]

        if len(cp_files) > 0:
            latest_file = max(cp_files, key=os.path.getctime)

            try:
                model.load_weights(latest_file)
            except Exception as e:
                __logger.exception(e)

    return model


def make_model(input_shape=(256, 256, 1), output_shape=63):  # default output shape is 3*21 for 21 joints in a hand
    encoder = tf.keras.applications.MobileNetV2(input_shape=input_shape,
                                                include_top=False,
                                                weights=None)  # 'imagenet')
    # encoder.trainable = False

    model = models.Sequential()
    model.add(encoder)
    model.add(layers.Flatten())
    model.add(layers.Dense(2048, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(output_shape, activation='relu'))
    model.summary()

    model = __try_load_checkpoint(model, __checkpoint_dir)
    return model


def train_model(model, train_data, batch_size, max_epochs, validation_gen=None):
    model.compile(optimizer=keras.optimizers.Adam(0.0001), loss=keras.losses.mean_squared_error, metrics=["mae", "acc"])

    if not os.path.exists(__checkpoint_dir): os.makedirs(__checkpoint_dir)
    if not os.path.exists(__tensorboard_dir): os.makedirs(__tensorboard_dir)

    checkpointer = keras.callbacks.ModelCheckpoint(
            filepath=str(__checkpoint_dir + __checkpoint_file_prefix + "weights.{epoch:02d}.hdf5"),
            save_best_only=False)
    tensorboard = keras.callbacks.TensorBoard(log_dir=__tensorboard_dir, histogram_freq=0,
                                              write_graph=True, write_images=True, update_freq='batch', profile_batch=0)
    prog = keras.callbacks.ProgbarLogger(count_mode='steps')
    model.fit(
            train_data,
            epochs=max_epochs,
            verbose=2,
            callbacks=[checkpointer, tensorboard, prog])


def estimate_pose(model, depth_img):
    return model.predict(depth_img).flatten()
