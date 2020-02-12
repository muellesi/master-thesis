import os

import tensorflow as tf



def _make_ae_head(input_shape = None):
    decoder = tf.keras.Sequential(name = 'decoder')
    decoder.add(tf.keras.layers.Conv2DTranspose(128, [3, 3],
                                                strides = 2,
                                                padding = 'same',
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


def _make_2d_pose_head_scale(num_output_channels = 21, end_sigmoid = True):
    model = tf.keras.Sequential(name = '2d-pose-scale')
    model.add(tf.keras.layers.Conv2D(64, [3, 3],
                                     padding = 'same',
                                     kernel_initializer = "glorot_normal",
                                     activation = 'relu'))
    model.add(tf.keras.layers.UpSampling2D((2, 2),
                                           interpolation = 'bilinear'))

    model.add(tf.keras.layers.Conv2D(64, [5, 5],
                                     padding = 'same',
                                     kernel_initializer = "glorot_normal",
                                     activation = 'relu'))
    model.add(tf.keras.layers.UpSampling2D((4, 4),
                                           interpolation = 'bilinear'))

    model.add(tf.keras.layers.Conv2D(32, [5, 5],
                                     padding = 'same',
                                     kernel_initializer = "glorot_normal",
                                     activation = 'relu'))
    model.add(tf.keras.layers.UpSampling2D((2, 2),
                                           interpolation = 'bilinear'))

    model.add(tf.keras.layers.Conv2D(num_output_channels, [5, 5],
                                     padding = 'same',
                                     kernel_initializer = "glorot_normal"
                                     ))
    model.add(tf.keras.layers.UpSampling2D((2, 2),
                                           interpolation = 'bilinear'))
    if end_sigmoid:
        model.add(tf.keras.layers.Activation(tf.keras.activations.sigmoid))

    return model


def _make_2D_pose_head(num_output_channels = 21, num_layers = 4, end_sigmoid = True):
    model = tf.keras.Sequential(name = '2d-pose')
    model.add(tf.keras.layers.Conv2DTranspose(64, [3, 3],
                                              strides = 2,
                                              padding = 'same',
                                              kernel_initializer = "glorot_normal",
                                              activation = 'relu'))
    if num_layers == 4:
        model.add(tf.keras.layers.Conv2DTranspose(64, [5, 5],
                                                  strides = 4,
                                                  padding = 'same',
                                                  kernel_initializer = "glorot_normal",
                                                  activation = 'relu'))
    elif num_layers == 5:
        model.add(tf.keras.layers.Conv2DTranspose(64, [5, 5],
                                                  strides = 2,
                                                  padding = 'same',
                                                  kernel_initializer = "glorot_normal",
                                                  activation = 'relu'))

        model.add(tf.keras.layers.Conv2DTranspose(64, [5, 5],
                                                  strides = 2,
                                                  padding = 'same',
                                                  kernel_initializer = "glorot_normal",
                                                  activation = 'relu'))

    model.add(tf.keras.layers.Conv2DTranspose(32, [5, 5],
                                              strides = 2,
                                              padding = 'same',
                                              kernel_initializer = "glorot_normal",
                                              activation = 'relu'))

    model.add(tf.keras.layers.Conv2DTranspose(num_output_channels, [5, 5],
                                              strides = 2,
                                              padding = 'same',
                                              kernel_initializer = "glorot_normal"
                                              ))
    if end_sigmoid:
        model.add(tf.keras.layers.Activation(tf.keras.activations.sigmoid))
    else:
        model.add(tf.keras.layers.Activation(tf.keras.activations.relu))
    return model


def _make_3D_pose_head(num_output_channels = 21):
    model = tf.keras.Sequential(name = '3d-pose')
    model.add(tf.keras.layers.Flatten())

    model.add(tf.keras.layers.Dense(1024, activation = 'relu'))
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Dense(1024, activation = 'relu'))
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Dense(30))

    model.add(tf.keras.layers.Dense(num_output_channels * 3))
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

    elif type == '3d-pose-est':
        return make_pose_estimator_3d(**kwargs)

    elif type == 'ae-head':
        model = _make_ae_head(**kwargs)

    else:
        raise ValueError("Found no model with name {}!".format(type))

    return model


def train_model(model,
                train_data,
                validation_data = None,
                max_epochs = None,
                learning_rate = 0.0001,
                tensorboard_dir = 'logs',
                do_clean_tensorboard_dir = True,
                checkpoint_dir = 'checkpoints',
                save_best_cp_only = False,
                best_cp_metric = 'val_loss',
                cp_name = 'cp_',
                cp_save_freq = 'epoch',
                loss = tf.keras.losses.mean_squared_error,
                use_lr_reduce = True,
                lr_reduce_factor = 0.75,
                lr_reduce_patience = 3,
                lr_reduce_min_lr = 0.00000001,
                lr_reduce_metric = 'loss',
                use_early_stop = True,
                early_stop_patience = 10,
                early_stop_metric = 'val_loss',
                verbose = 1,
                add_metrics = None,
                steps_per_epoch = None,
                validation_steps = None,
                custom_callbacks = None,
                optimizer = 'adam',
                optimizer_clipnorm = None,
                optimizer_clipvalue = None,
                optimizer_momentum = None):

    opt_add_args = {}
    if optimizer_clipnorm:
        opt_add_args["clipnorm"] = optimizer_clipnorm

    if optimizer_clipvalue:
        opt_add_args["clipvalue"] = optimizer_clipvalue

    if optimizer_momentum:
        opt_add_args["momentum"] = optimizer_momentum

    if optimizer == 'adam':
        print("Using optimizer {}".format("Adam"))
        optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate, **opt_add_args)
    if optimizer == 'sgd':
        print("Using optimizer {} with momentum {}".format("SGD", optimizer_momentum))
        optimizer = tf.keras.optimizers.SGD(learning_rate = learning_rate, **opt_add_args)

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
            save_best_only = save_best_cp_only,
            monitor = best_cp_metric,
            save_freq = cp_save_freq
            )
    callbacks.append(checkpointer)

    tensorboard = tf.keras.callbacks.TensorBoard(log_dir = tensorboard_dir,
                                                 histogram_freq = 0,
                                                 write_graph = True,
                                                 write_images = True,
                                                 update_freq = 'batch',
                                                 profile_batch = 0)
    callbacks.append(tensorboard)

    if use_lr_reduce:
        lr_reduce = tf.keras.callbacks.ReduceLROnPlateau(monitor = lr_reduce_metric,
                                                         factor = lr_reduce_factor,
                                                         patience = lr_reduce_patience,
                                                         verbose = 1,
                                                         min_lr = lr_reduce_min_lr)
        callbacks.append(lr_reduce)

    if use_early_stop:
        early_stop = tf.keras.callbacks.EarlyStopping(monitor = early_stop_metric,
                                                      patience = early_stop_patience,
                                                      verbose = 1,
                                                      restore_best_weights = True)
        callbacks.append(early_stop)

    if custom_callbacks is not None:
        callbacks.extend(custom_callbacks)

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
                           regressor_weights = None,
                           num_head_layers = 4,
                           end_sigmoid = True,
                           use_transpose_conv = True):
    inputs = tf.keras.Input(shape = input_shape)

    if encoder_weights is not None:
        encoder = tf.keras.models.load_model(encoder_weights)
    else:
        encoder = make_model('mobilenetv2',
                             input_shape = input_shape,
                             include_top = False,
                             weights = None
                             )
    encoder.trainable = True

    latent = encoder(inputs)

    if use_transpose_conv:
        regressor = _make_2D_pose_head(num_output_channels = num_joints,
                                       num_layers = num_head_layers,
                                       end_sigmoid = end_sigmoid)
    else:
        regressor = _make_2d_pose_head_scale(num_output_channels = num_joints,
                                             end_sigmoid = end_sigmoid)

    if regressor_weights is not None:
        regressor = regressor.load_weights(regressor_weights)

    predictions = regressor(latent)

    model = tf.keras.Model(inputs = inputs, outputs = predictions)
    return model


def make_pose_estimator_3d(input_shape,
                           num_joints = 21,
                           encoder_weights = None,
                           regressor_weights = None,
                           encoder_type = 'mobilenet'):
    inputs = tf.keras.Input(shape = input_shape)

    if encoder_type == 'mobilenet':
        if encoder_weights is not None:
            encoder = tf.keras.models.load_model(encoder_weights)
        else:
            encoder = make_model('mobilenetv2',
                                 input_shape = input_shape,
                                 include_top = False,
                                 weights = None)

        encoder.trainable = True
        latent = encoder(inputs)
    else:
        x = tf.keras.layers.Conv2D(8,
                                   (5, 5),
                                   padding = 'same',
                                   activation = tf.keras.activations.relu,
                                   kernel_regularizer = tf.keras.regularizers.l1_l2(0.001)
                                   )(inputs)
        x = tf.keras.layers.MaxPool2D(4, 4)(x)

        x = tf.keras.layers.Conv2D(8,
                                   (5, 5),
                                   padding = 'same',
                                   activation = tf.keras.activations.relu,
                                   kernel_regularizer = tf.keras.regularizers.l1_l2(0.001)
                                   )(x)
        x = tf.keras.layers.MaxPool2D(2, 2)(x)

        x = tf.keras.layers.Conv2D(8,
                                   (3, 3),
                                   padding = 'same',
                                   activation = tf.keras.activations.relu,
                                   kernel_regularizer = tf.keras.regularizers.l1_l2(0.001)
                                   )(x)

        x = tf.keras.layers.Conv2D(8,
                                   (5, 5),
                                   padding = 'same',
                                   activation = tf.keras.activations.relu,
                                   kernel_regularizer = tf.keras.regularizers.l1_l2(0.001)
                                   )(x)
        x = tf.keras.layers.MaxPool2D(2, 2)(x)

        x = tf.keras.layers.Conv2D(8,
                                   (5, 5),
                                   padding = 'same',
                                   activation = tf.keras.activations.relu,
                                   kernel_regularizer = tf.keras.regularizers.l1_l2(0.001)
                                   )(x)
        x = tf.keras.layers.MaxPool2D(2, 2)(x)

        x = tf.keras.layers.Conv2D(8,
                                   (3, 3),
                                   padding = 'same',
                                   activation = tf.keras.activations.relu,
                                   kernel_regularizer = tf.keras.regularizers.l1_l2(0.001)
                                   )(x)

        x = tf.keras.layers.Conv2D(8,
                                   (5, 5),
                                   padding = 'same',
                                   activation = tf.keras.activations.relu,
                                   kernel_regularizer = tf.keras.regularizers.l1_l2(0.001)
                                   )(x)
        x = tf.keras.layers.MaxPool2D(2, 2)(x)

        x = tf.keras.layers.Conv2D(8,
                                   (5, 5),
                                   padding = 'same',
                                   activation = tf.keras.activations.relu,
                                   kernel_regularizer = tf.keras.regularizers.l1_l2(0.001)
                                   )(x)
        latent = tf.keras.layers.Conv2D(8,
                                        (3, 3),
                                        padding = 'same',
                                        activation = tf.keras.activations.relu,
                                        kernel_regularizer = tf.keras.regularizers.l1_l2(0.001)
                                        )(x)

    regressor = _make_3D_pose_head(num_output_channels = num_joints)

    if regressor_weights is not None:
        regressor = regressor.load_weights(regressor_weights)

    predictions = regressor(latent)

    model = tf.keras.Model(inputs = inputs, outputs = predictions)
    return model
