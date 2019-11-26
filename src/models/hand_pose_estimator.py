import tensorflow as tf
import numpy as np
import tools
import os
import math
import models.model_helpers as mh

__checkpoint_dir = os.path.join("pose_est", "checkpoints")
__tensorboard_dir = os.path.join("pose_est", "tensorboard_logs")
__checkpoint_file_prefix = "pose_est_"
__logger = tools.get_logger(__name__, do_file_logging=False)


def make_model(input_shape=(256, 256, 1), output_shape=63, data_dir="data"):  # default output shape is 3*21 for 21 joints in a hand
    """
    Creates a new hand pose estimation model. If data_dir is not None: tries to load weights from latest checkpoint in data_dir/pose_est/checkpoints
    :return:        Keras sequential model for pose estimation
    """
    encoder = tf.keras.applications.MobileNetV2(input_shape=input_shape,
                                                include_top=False,
                                                weights=None)  # 'imagenet')
    encoder.trainable = True

    model = tf.keras.models.Sequential()
    model.add(encoder)
    model.add(tf.keras.layers.Flatten())

    # FC 1
    model.add(tf.keras.layers.Dense(2048,
                                    kernel_initializer=tf.keras.initializers.GlorotNormal(),
                                    bias_initializer='zeros'))
    model.add(tf.keras.layers.Activation('relu'))
    # model.add(tf.keras.layers.Dropout(0.2))

    # FC 2
    model.add(tf.keras.layers.Dense(1024,
                                    kernel_initializer=tf.keras.initializers.GlorotNormal(),
                                    bias_initializer='zeros'))
    model.add(tf.keras.layers.Activation('relu'))
    # model.add(tf.keras.layers.Dropout(0.2))

    # FC 3 - Bottleneck
    # model.add(tf.keras.layers.Dense(40))
    # model.add(tf.keras.layers.Activation('relu'))
    # model.add(tf.keras.layers.Dropout(0.2))

    model.add(tf.keras.layers.Dense(output_shape,
                                    kernel_initializer=tf.keras.initializers.Orthogonal(gain=100.0),
                                    bias_initializer='zeros'))
    model.summary()

    if data_dir:
        model = mh.try_load_checkp oint(model=model,
                                       checkpoint_dir=os.path.join(data_dir, __checkpoint_dir),
                                       checkpoint_file_prefix=__checkpoint_file_prefix)

    return model


def train_model(model, train_data, max_epochs, learning_rate=0.0001, validation_data=None, data_dir="data"):
    tensorboard_dir = os.path.join(data_dir, __tensorboard_dir)
    checkpoint_dir = os.path.join(data_dir, __checkpoint_dir)

    tools.clean_tensorboard_logs(tensorboard_dir)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate,
                                         clipvalue=10)
    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.mean_squared_error,
                  metrics=["mae", "acc"])

    if not os.path.exists(checkpoint_dir): os.makedirs(checkpoint_dir)
    if not os.path.exists(tensorboard_dir): os.makedirs(tensorboard_dir)

    checkpointer = tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(checkpoint_dir, __checkpoint_file_prefix + "weights.{epoch:02d}.hdf5"),
            save_best_only=False)

    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_dir,
                                                 histogram_freq=0,
                                                 write_graph=True,
                                                 write_images=True,
                                                 update_freq='batch',
                                                 profile_batch=0)


    def scheduler(epoch):
        if epoch < 10:
            tf.print("Learning rate in epoch ", epoch, ": ", learning_rate)
            return float(learning_rate)
        else:
            lr = learning_rate * math.exp(0.1 * (10.0 - epoch))
            tf.print("Learning rate in epoch ", epoch, ": ", lr)
            return lr


    lr_decay = tf.keras.callbacks.LearningRateScheduler(scheduler)

    model.fit(
            train_data,
            validation_data=validation_data,
            epochs=max_epochs,
            verbose=2,
            callbacks=[checkpointer, tensorboard])  # no decay for now...


def estimate_pose(model, depth_img):
    return model.predict(depth_img).flatten()
