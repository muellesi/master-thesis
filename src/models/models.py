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
                                              kernel_initializer = "glorot_normal"
                                              ))

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
        return make_pose_estimator_2d(**kwargs)

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
                use_early_stop           = True,
                verbose                  = 1,
                add_metrics              = None,
                steps_per_epoch          = None,
                validation_steps         = None):

    if do_clean_tensorboard_dir:
        tools.clean_tensorboard_logs(tensorboard_dir)

    optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate,
                                         clipvalue = 10)

    metrics = ["mae", "acc"]
    if add_metrics is not None:
        metrics.extend(add_metrics)

    model.compile(optimizer = optimizer, loss = loss, metrics = metrics)

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

    fit_params = {
            'x'              : train_data,
            'validation_data': validation_data,
            'epochs'         : max_epochs,
            'verbose'        : verbose,
            'callbacks'      : callbacks
            }

    if steps_per_epoch is not None:
        fit_params['steps_per_epoch'] = steps_per_epoch

    if validation_steps is not None:
        fit_params['validation_steps'] = validation_steps

    model.fit(**fit_params)


def make_pose_estimator_2d(input_shape,
                           num_joints = 21,
                           encoder_weights = None,
                           regressor_weights = None):

    inputs = tf.keras.Input(shape = input_shape)

    if encoder_weights is not None:
        encoder = tf.keras.models.load_model(encoder_weights)
    else:
        encoder = make_model('mobilenetv2',
                                  input_shape = input_shape,
                                  include_top = False,
                                  weights     = None)
    encoder.trainable = True

    latent = encoder(inputs)

    if regressor_weights is not None:
        regressor = tf.keras.models.load_model(regressor_weights)
    else:
        regressor = _make_2D_pose_head(num_output_channels = num_joints)

    predictions = regressor(latent)

    model = tf.keras.Model(inputs = inputs, outputs = predictions)
    return model
