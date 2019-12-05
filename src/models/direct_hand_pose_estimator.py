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


def make_model(input_shape=(256, 256, 1),
               output_shape=63,  # default output shape is 3*21 for 21 joints in a hand
               data_dir="data",
               path_encoder_pretrained=None,
               encoder_trainable=True,
               path_decoder_pretrained=None):
    """
    Creates a new hand pose estimation model. If data_dir is not None: tries to load weights from latest checkpoint in data_dir/pose_est/checkpoints
    :return:        Keras sequential model for pose estimation
    """
    if path_encoder_pretrained:
        encoder = tf.keras.Model.load(path_encoder_pretrained)
    else:
        encoder = tf.keras.applications.MobileNetV2(input_shape=input_shape,
                                                    include_top=False,
                                                    weights=None)
    encoder.trainable = encoder_trainable

    regressor = tf.keras.models.Sequential()
    regressor.add(tf.keras.layers.Flatten(input_shape=encoder.output_shape[1:]))

    # FC 1
    regressor.add(tf.keras.layers.Dense(2048,
                                        kernel_initializer=tf.keras.initializers.GlorotNormal(),
                                        bias_initializer='zeros'))
    regressor.add(tf.keras.layers.Activation('relu'))
    # model.add(tf.keras.layers.Dropout(0.2))

    # FC 2
    regressor.add(tf.keras.layers.Dense(1024,
                                        kernel_initializer=tf.keras.initializers.GlorotNormal(),
                                        bias_initializer='zeros'))
    regressor.add(tf.keras.layers.Activation('relu'))
    # model.add(tf.keras.layers.Dropout(0.2))

    # FC 3 - Bottleneck
    # model.add(tf.keras.layers.Dense(40))
    # model.add(tf.keras.layers.Activation('relu'))
    # model.add(tf.keras.layers.Dropout(0.2))

    regressor.add(tf.keras.layers.Dense(output_shape,
                                        kernel_initializer=tf.keras.initializers.Orthogonal(gain=100.0),
                                        bias_initializer='zeros'))
    regressor.summary()

    model = tf.keras.models.Sequential()
    model.add(encoder)
    model.add(regressor)

    if data_dir:
        model = mh.try_load_checkpoint(model=model,
                                       checkpoint_dir=os.path.join(data_dir, __checkpoint_dir),
                                       checkpoint_file_prefix=__checkpoint_file_prefix)

    return model





def estimate_pose(model, depth_img):
    return model.predict(depth_img).flatten()
