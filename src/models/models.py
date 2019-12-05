import tensorflow as tf
import tools



def _make_ae_head(input_shape = None):
    decoder = tf.keras.Sequential(name = 'decoder')
    decoder.add(tf.keras.layers.Conv2DTranspose(128, [3, 3], strides = 2, padding = 'same', input_shape = input_shape))
    decoder.add(tf.keras.layers.Activation(tf.keras.activations.tanh))

    decoder.add(tf.keras.layers.Conv2DTranspose(64, [5, 5], strides = 2, padding = 'same'))
    decoder.add(tf.keras.layers.Activation(tf.keras.activations.tanh))

    decoder.add(tf.keras.layers.Conv2DTranspose(16, [5, 5], strides = 2, padding = 'same'))
    decoder.add(tf.keras.layers.Activation(tf.keras.activations.tanh))

    decoder.add(tf.keras.layers.Conv2DTranspose(16, [5, 5], strides = 2, padding = 'same'))
    decoder.add(tf.keras.layers.Activation(tf.keras.activations.tanh))

    decoder.add(tf.keras.layers.Conv2DTranspose(1, [5, 5], strides = 2, padding = 'same'))
    decoder.add(tf.keras.layers.Activation(tf.keras.activations.sigmoid))

    return decoder


def _make_pose_head_simple(input_shape = None):
    decoder = tf.keras.Sequential(name = 'decoder')
    decoder.add(tf.keras.layers.Conv2DTranspose(128, [3, 3], strides = 2, padding = 'same', input_shape = input_shape, kernel_initializer="glorot_normal"))

    decoder.add(tf.keras.layers.Conv2DTranspose(64, [5, 5], strides = 2, padding = 'same', kernel_initializer="glorot_normal"))

    decoder.add(tf.keras.layers.Conv2DTranspose(16, [5, 5], strides = 2, padding = 'same', kernel_initializer="glorot_normal"))

    decoder.add(tf.keras.layers.Conv2DTranspose(16, [5, 5], strides = 2, padding = 'same', kernel_initializer="glorot_normal"))

    decoder.add(tf.keras.layers.Conv2DTranspose(1, [5, 5], strides = 2, padding = 'same', kernel_initializer="glorot_normal"))
    decoder.add(tf.keras.layers.Activation(tf.keras.activations.linear))
    

def make_model(type: str, **kwargs):
    model = None

    if type == 'mobilenetv2':
        model = tf.keras.applications.MobileNetV2(**kwargs)

    elif type == 'vgg16':
        model = tf.keras.applications.vgg16.VGG16(**kwargs)

    elif type == 'pose-est-head-simple':
        return _make_pose_head_simple(**kwargs)

    elif type == 'ae-head':
        model = _make_ae_head(**kwargs)


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
                cp_name                  = 'cp_'
                ):

    if do_clean_tensorboard_dir:
        tools.clean_tensorboard_logs(tensorboard_dir)

    optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate,
                                         clipvalue = 10)
    model.compile(optimizer = optimizer,
                  loss    = tf.keras.losses.mean_squared_error,
                  metrics = ["mae", "acc"])

    if not os.path.exists(checkpoint_dir): os.makedirs(checkpoint_dir)
    if not os.path.exists(tensorboard_dir): os.makedirs(tensorboard_dir)

    checkpointer = tf.keras.callbacks.ModelCheckpoint(
            filepath = os.path.join(checkpoint_dir, cp_name + ".{epoch:02d}" + ".hdf5"),
            save_best_only = False)

    tensorboard = tf.keras.callbacks.TensorBoard(log_dir = tensorboard_dir,
                                                 histogram_freq = 0,
                                                 write_graph    = True,
                                                 write_images   = True,
                                                 update_freq    = 'batch',
                                                 profile_batch  = 0)


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
            validation_data = validation_data,
            epochs          = max_epochs,
            verbose         = 2,
            callbacks       = [checkpointer, tensorboard])  # no decay for now...
