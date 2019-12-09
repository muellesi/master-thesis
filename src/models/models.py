import os

import tensorflow as tf

import tools



def _make_ae_head(input_shape = None):
    decoder = tf.keras.Sequential(name = 'decoder')
    decoder.add(tf.keras.layers.Conv2DTranspose(128, [3, 3],
                                                strides     = 2,
                                                padding     = 'same',
                                                input_shape = input_shape))
    decoder.add(tf.keras.layers.Activation(tf.keras.activations.tanh))

    decoder.add(tf.keras.layers.Conv2DTranspose(64, [5, 5],
                                                strides = 2,
                                                padding = 'same'))
    decoder.add(tf.keras.layers.Activation(tf.keras.activations.tanh))

    decoder.add(tf.keras.layers.Conv2DTranspose(16, [5, 5],
                                                strides = 2,
                                                padding = 'same'))
    decoder.add(tf.keras.layers.Activation(tf.keras.activations.tanh))

    decoder.add(tf.keras.layers.Conv2DTranspose(16, [5, 5],
                                                strides = 2,
                                                padding = 'same'))
    decoder.add(tf.keras.layers.Activation(tf.keras.activations.tanh))

    decoder.add(tf.keras.layers.Conv2DTranspose(1, [5, 5],
                                                strides = 2,
                                                padding = 'same'))
    decoder.add(tf.keras.layers.Activation(tf.keras.activations.sigmoid))

    return decoder


def _make_2D_pose_head(num_output_channels = 21):
    model = tf.keras.Sequential(name = '2d-pose')
    model.add(tf.keras.layers.Conv2DTranspose(128, [3, 3],
                                              strides            = 2,
                                              padding            = 'same',
                                              kernel_initializer = "glorot_normal",
                                              activation         = 'relu'))

    model.add(tf.keras.layers.Conv2DTranspose(64, [3, 3],
                                              strides            = 2,
                                              padding            = 'same',
                                              kernel_initializer = "glorot_normal",
                                              activation         = 'relu'))

    model.add(tf.keras.layers.Conv2DTranspose(32, [3, 3],
                                              strides            = 2,
                                              padding            = 'same',
                                              kernel_initializer = "glorot_normal",
                                              activation         = 'relu'))

    model.add(tf.keras.layers.Conv2DTranspose(32, [3, 3],
                                              strides            = 2,
                                              padding            = 'same',
                                              kernel_initializer = "glorot_normal",
                                              activation         = 'relu'))

    model.add(tf.keras.layers.Conv2DTranspose(num_output_channels, [5, 5],
                                              strides            = 2,
                                              padding            = 'same',
                                              kernel_initializer = "glorot_normal",
                                              activation         = 'relu'))

    model.add(tf.keras.layers.Activation(tf.keras.activations.linear))

    return model


def make_model(type: str, **kwargs):
    model = None

    if type == 'mobilenetv2':
        # input_shape, include_top, weights
        model = tf.keras.applications.MobileNetV2(**kwargs)

    elif type == 'vgg16':
        model = tf.keras.applications.vgg16.VGG16(**kwargs)

    elif type == '2d-pose-est':
        return PoseEstimator2D(**kwargs)

    elif type == 'ae-head':
        model = _make_ae_head(**kwargs)

    else:
        raise ValueError("Found no model with name {}!".format(type))

    return model


def train_model(model, 
                train_data,
                validation_data          = None,
                max_epochs               = None,
                learning_rate            = 0.0001,
                tensorboard_dir          = 'logs',
                do_clean_tensorboard_dir = True,
                checkpoint_dir           = 'checkpoints',
                save_best_cp_only        = False,
                best_cp_metric           = 'val_acc',
                cp_name                  = 'cp_',
                loss                     = tf.keras.losses.mean_squared_error,
                use_lr_reduce            = True,
                use_early_stop           = True):

    if do_clean_tensorboard_dir:
        tools.clean_tensorboard_logs(tensorboard_dir)

    optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate,
                                         clipvalue = 10)

    model.compile(optimizer = optimizer, loss = loss, metrics = ["mae", "acc"])

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)

    callbacks = []

    checkpointer = tf.keras.callbacks.ModelCheckpoint(
            filepath = os.path.join(checkpoint_dir,
                                    cp_name + ".{epoch:02d}" + ".hdf5"),
            save_best_only = False)
    callbacks.append(checkpointer)

    tensorboard = tf.keras.callbacks.TensorBoard(log_dir        = tensorboard_dir,
                                                 histogram_freq = 0,
                                                 write_graph    = True,
                                                 write_images   = True,
                                                 update_freq    = 'batch',
                                                 profile_batch  = 0)
    callbacks.append(tensorboard)

    if use_lr_reduce:
        lr_reduce = tf.keras.callbacks.ReduceLROnPlateau(monitor  = 'val_loss',
                                                         factor   = 0.75,
                                                         patience = 3,
                                                         verbose  = 1,
                                                         min_lr   = 0.00000001)
        callbacks.append(lr_reduce)

    if use_early_stop:
        early_stop = tf.keras.callbacks.EarlyStopping(monitor              = 'val_loss',
                                                      patience             = 5,
                                                      verbose              = 1,
                                                      restore_best_weights = True)
        callbacks.append(early_stop)

    model.fit(x = train_data,
              validation_data = validation_data,
              epochs          = max_epochs,
              verbose         = 2,
              callbacks       = callbacks)


class PoseEstimator2D(tf.keras.Model):

    def __init__(self, input_shape, num_joints = 21, encoder_weights = None,
                 regressor_weights = None):

        super(PoseEstimator2D, self).__init__()

        if encoder_weights is not None:
            self.encoder = tf.keras.models.load_model(encoder_weights)
        else:
            self.encoder = make_model('mobilenetv2',
                                      input_shape = input_shape,
                                      include_top = False,
                                      weights     = None)

        if regressor_weights is not None:
            self.regressor = tf.keras.models.load_model(regressor_weights)
        else:
            self.regressor = _make_2D_pose_head(num_output_channels = num_joints)

        self.encoder.trainable = True


    def set_encoder_trainable(self, encoder_trainable: bool):
        self.encoder.trainable = encoder_trainable


    def call(self, inputs, training = False):
        latent    = self.encoder(inputs)
        conf_maps = self.regressor(latent)

        return conf_maps
