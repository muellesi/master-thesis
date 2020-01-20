import hashlib
import json
import os
import random
import shutil
import time
from itertools import product

import tensorflow as tf
from google_drive_downloader import GoogleDriveDownloader as gdd
from tensorboard.plugins.hparams import api as hp

import datasets.util
import tools.training_callbacks
from datasets import SerializedDataset
from datasets.tfrecord_helper import decode_confmaps
from datasets.tfrecord_helper import depth_and_confmaps
from tools.training_callbacks.telegram_callback import TelegramCallback
import pprint


logger = tools.get_logger('optimize_2d_pose_est', do_file_logging = True)


def prepare_ds(name, ds, add_noise, add_empty, augment, hparams):
    ds = ds.map(depth_and_confmaps,
                num_parallel_calls = tf.data.experimental.AUTOTUNE)

    ds = ds.map(lambda img, confm: (img, decode_confmaps(confm)),
                num_parallel_calls = tf.data.experimental.AUTOTUNE)

    ds = ds.map(lambda img, confm:
                (datasets.util.scale_clip_image_data(img, 1.0 / 2500.0),
                 datasets.util.scale_clip_image_data(confm, 1.0 / 2 ** 16)),
                num_parallel_calls = tf.data.experimental.AUTOTUNE)

    ds = ds.map(lambda img, confm: (img,
                                    confm *
                                    (tf.math.divide_no_nan(
                                            tf.constant(1.0,
                                                        dtype =
                                                        tf.dtypes.float32),
                                            tf.reduce_max(confm)))),
                num_parallel_calls = tf.data.experimental.AUTOTUNE)

    if add_noise:
        ds = ds.map(lambda img, confm:
                    (datasets.util.add_random_noise(img),
                     confm),
                    num_parallel_calls = tf.data.experimental.AUTOTUNE)

    if augment:
        ds = ds.map(lambda img, confm: datasets.util.augment_depth_and_confmaps(img, confm, hparams[HP_AUGMENTATION]),
                    num_parallel_calls = tf.data.experimental.AUTOTUNE)

    if add_empty:
        ds_empty_imgs = datasets.util.make_img_ds_from_glob(
                empty_background_path,
                width = net_input_width,
                height = net_input_height,
                value_scale = 1.0 / 2500.0,
                shuffle = True)
        ds_empty_imgs = ds_empty_imgs.map(lambda img: (
                img, tf.zeros(
                        [net_input_height, net_input_width, num_skel_joints])))
        ds = ds.concatenate(ds_empty_imgs)

    return ds


def batched_twod_argmax(val):
    maxy = tf.argmax(tf.reduce_max(val, axis = 2), 1)
    maxx = tf.argmax(tf.reduce_max(val, axis = 1), 1)
    maxs = tf.stack([maxy, maxx], axis = 2)
    maxs = tf.cast(maxs, dtype = tf.dtypes.float32)
    return maxs


def keypoint_error_metric(y_true, y_pred):
    dist = batched_twod_argmax(y_true) - batched_twod_argmax(y_pred)
    dist = tf.norm(dist, axis = 2)
    mean_dists = tf.reduce_mean(dist, axis = 0)
    return mean_dists


def make_skewed_mse(asymmetry_factor):
    def acost(y_true, y_pred):
        return tf.pow(y_pred - y_true, 2) * tf.pow(
                tf.sign(y_pred - y_true) + asymmetry_factor, 2)

    return acost


def make_kpe_mse(kpe_weight):
    def kpe_mse(y_true, y_pred):
        res = tf.keras.losses.mean_squared_error(y_true, y_pred) + kpe_weight * keypoint_error_metric(y_true, y_pred)
        return res
    return kpe_mse

def make_model(hparams):
    input_shape = [224, 224, 1]
    num_output_channels = 21

    dropout_ratio = hparams[HP_DROPOUT] / 100

    regularizer = None
    if hparams[HP_L1REGULARIZATION] != 0 or hparams[HP_L2REGULARIZATION] != 0:
        regularizer = tf.keras.regularizers.l1_l2(l1 = hparams[HP_L1REGULARIZATION] / 100,
                                                  l2 = hparams[HP_L2REGULARIZATION] / 100)

    inputs = tf.keras.Input(shape = input_shape)

    encoder = tf.keras.models.load_model(encoder_pretrained_path)
    encoder.trainable = True
    latent = encoder(inputs)

    regressor = tf.keras.Sequential(name = '2d-pose')

    regressor.add(tf.keras.layers.Conv2DTranspose(128, [3, 3],
                                                  strides = 2,
                                                  padding = 'same',
                                                  kernel_initializer = "glorot_normal",
                                                  kernel_regularizer = regularizer,
                                                  activation = 'relu'))

    if dropout_ratio > 0:
        regressor.add(tf.keras.layers.Dropout(dropout_ratio))

    if hparams[HP_DECODERLAYERS] == 4:
        regressor.add(tf.keras.layers.Conv2DTranspose(64, [5, 5],
                                                      strides = 4,
                                                      padding = 'same',
                                                      kernel_initializer = "glorot_normal",
                                                      kernel_regularizer = regularizer,
                                                      activation = 'relu'))

        if dropout_ratio > 0:
            regressor.add(tf.keras.layers.Dropout(dropout_ratio))

    elif hparams[HP_DECODERLAYERS] == 5:
        regressor.add(tf.keras.layers.Conv2DTranspose(64, [5, 5],
                                                      strides = 2,
                                                      padding = 'same',
                                                      kernel_initializer = "glorot_normal",
                                                      kernel_regularizer = regularizer,
                                                      activation = 'relu'))
        if dropout_ratio > 0:
            regressor.add(tf.keras.layers.Dropout(dropout_ratio))

        regressor.add(tf.keras.layers.Conv2DTranspose(64, [5, 5],
                                                      strides = 2,
                                                      padding = 'same',
                                                      kernel_initializer = "glorot_normal",
                                                      kernel_regularizer = regularizer,
                                                      activation = 'relu'))

    regressor.add(tf.keras.layers.Conv2DTranspose(32, [5, 5],
                                                  strides = 2,
                                                  padding = 'same',
                                                  kernel_initializer = "glorot_normal",
                                                  kernel_regularizer = regularizer,
                                                  activation = 'relu'))

    regressor.add(tf.keras.layers.Conv2DTranspose(num_output_channels, [5, 5],
                                                  strides = 2,
                                                  padding = 'same',
                                                  kernel_initializer = "glorot_normal"
                                                  ))

    regressor.add(tf.keras.layers.Activation(tf.keras.activations.sigmoid))

    predictions = regressor(latent)

    model = tf.keras.Model(inputs = inputs, outputs = predictions)
    model.build(input_shape = input_shape)
    return model


def train_pose_estimator(model, train_data, validation_data, test_data, hparams, run_id, tensorboard_dir,
                         checkpoint_dir):
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    learning_rate = hparams[HP_LEARNINGRATE] / 10 ** 5
    max_epochs = 100

    optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate, clipvalue = 10)

    if hparams[HP_LOSS] == "skewed":
        loss = make_skewed_mse(-0.5)
    else:
        loss = hparams[HP_LOSS]

    metrics = ["mae", "accuracy", keypoint_error_metric]

    # Begin with transfer learning if configured
    if hparams[HP_PRETRAINING]:
        logger.info('Pretraining: Disabling training for encoder')
        model.get_layer(index = 1).trainable = False
        model.compile(optimizer = optimizer, loss = loss, metrics = metrics)
        model.summary(print_fn = logger.info)

        tensorboard = tf.keras.callbacks.TensorBoard(log_dir = tensorboard_dir,
                                                     histogram_freq = 0,
                                                     update_freq = 'batch',
                                                     profile_batch = 0)  # workaround for issue #2084

        checkpoint_name = "pretraining-epoch-{epoch:02d}" + ".hdf5"
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath = os.path.join(checkpoint_dir, checkpoint_name),
                save_best_only = True,
                monitor = 'val_loss'
                )

        lr_reduce_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor = "loss",
                                                                  factor = 0.75,
                                                                  patience = 2,
                                                                  verbose = 1,
                                                                  min_lr = 0.0000005)

        early_stop_callback = tf.keras.callbacks.EarlyStopping(monitor = "val_loss",
                                                               patience = 5,
                                                               verbose = 1,
                                                               restore_best_weights = True)

        hparams_callback = hp.KerasCallback(tensorboard_dir, hparams, trial_id=str(run_id))

        telegram_callback = TelegramCallback(
                user = telegram_user,
                token = telegram_token,
                chat_id = telegram_chat,
                training_id = "TransferLearning" + str(run_id),
                training_description = pprint.pformat(hparams)
                )

        callbacks = [lr_reduce_callback, early_stop_callback, checkpoint_callback, tensorboard, telegram_callback,
                     hparams_callback]

        fit_params = {
                'x'              : train_data,
                'validation_data': validation_data,
                'epochs'         : max_epochs,
                'callbacks'      : callbacks
                }

        logger.info("Starting pretraining...")
        model.fit(**fit_params)

        logger.info("Pretraining done.")

        # Save end result
        os.makedirs(os.path.join(checkpoint_dir, "final"), exist_ok = True)
        final_save_name = os.path.join(checkpoint_dir, "final", "prelim_end-" + str(run_id) + ".hdf5")
        logger.info("Saving model as {}".format(final_save_name))
        model.save(filepath = final_save_name, overwrite = True, include_optimizer = False, save_format = 'h5')

    # Main Training
    logger.info('Enabling training for encoder for fine tuning')
    model.get_layer(index = 1).trainable = True
    model.compile(optimizer = optimizer, loss = loss, metrics = metrics)
    model.summary(print_fn = logger.info)

    tensorboard = tf.keras.callbacks.TensorBoard(log_dir = tensorboard_dir,
                                                 histogram_freq = 0,
                                                 update_freq = 'batch',
                                                 profile_batch = 0)  # workaround for issue #2084

    checkpoint_name = "main-training-epoch-{epoch:02d}" + ".hdf5"
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath = os.path.join(checkpoint_dir, checkpoint_name),
            save_best_only = True,
            monitor = 'val_loss'
            )

    lr_reduce_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor = "loss",
                                                              factor = 0.75,
                                                              patience = 2,
                                                              verbose = 1,
                                                              min_lr = 0.0000005)

    early_stop_callback = tf.keras.callbacks.EarlyStopping(monitor = "val_loss",
                                                           patience = 5,
                                                           verbose = 1,
                                                           restore_best_weights = True)

    hparams_callback = hp.KerasCallback(tensorboard_dir, hparams, trial_id=str(run_id))

    telegram_callback = TelegramCallback(
            user = telegram_user,
            token = telegram_token,
            chat_id = telegram_chat,
            training_id = "MainTraining" + str(run_id),
            training_description = pprint.pformat(hparams)
            )

    callbacks = [lr_reduce_callback, early_stop_callback, checkpoint_callback, tensorboard, telegram_callback,
                 hparams_callback]

    fit_params = {
            'x'              : train_data,
            'validation_data': validation_data,
            'epochs'         : max_epochs,
            'callbacks'      : callbacks
            }

    logger.info("Starting main training...")
    model.fit(**fit_params)

    logger.info("Main training done.")

    # Save end result
    os.makedirs(os.path.join(checkpoint_dir, "final"), exist_ok = True)
    final_save_name = os.path.join(checkpoint_dir, "final", "main_end-" + str(run_id) + ".hdf5")
    logger.info("Saving model as {}".format(final_save_name))
    model.save(filepath = final_save_name, overwrite = True, include_optimizer = False, save_format = 'h5')

    logger.info("Starting test...")
    test_result = model.evaluate(
            x = test_data,
            verbose = 1
            )

    test_result_labeled = dict(zip(model.metrics_names, test_result))
    logger.info(test_result_labeled)


def run(run_configs):
    while len(run_configs) > 0:

        random_index = random.randint(0, len(run_configs) - 1)
        hparams = run_configs.pop(random_index)

        hparams_string = str(hparams)
        run_id = hashlib.sha256(hparams_string.encode("utf-8")).hexdigest()

        run_logdir = os.path.join(base_logdir, 'tensorboard', str(run_id))
        run_checkpoint_dir = os.path.join(base_logdir, 'checkpoints', str(run_id))

        remote_tensorboard_dir = os.path.join(logdir_remote, "tensorboard", str(run_id))
        remote_checkpoint_dir = os.path.join(logdir_remote, "checkpoints", str(run_id))

        # Create file with current run id in remote so other clients know this id is taken
        remote_registration_file = os.path.join(logdir_remote, "registration", str(run_id))

        if os.path.exists(remote_registration_file):
            continue  # skip this run since it is running elswhere/was already done

        registration_success = False
        while not registration_success:
            try:
                os.makedirs(os.path.dirname(remote_registration_file), exist_ok = True)
                open(remote_registration_file, 'a').close()
            except Exception as ex:
                logger.exception(ex)
            else:
                registration_success = True

        logger.info("---------------------------------------------------------")
        logger.info("Preparing training run {}".format(run_id))
        logger.info("Run configuration: {}".format(pprint.pformat(hparams)))
        logger.info("---------------------------------------------------------")

        ds_provider = SerializedDataset(ds_settings[selected_dataset], cache_path = dataset_cache_path)
        ds_train = ds_provider.get_data("train")
        ds_train = prepare_ds('train',
                              ds_train,
                              add_noise = True,
                              add_empty = True,
                              augment = True,
                              hparams = hparams)
        ds_train = datasets.util.batch_shuffle_prefetch(ds_train, batch_size = batch_size)

        ds_val = ds_provider.get_data("validation")
        ds_val = prepare_ds('validation',
                            ds_val,
                            add_noise = False,
                            add_empty = False,
                            augment = False,
                            hparams = hparams)
        ds_val = ds_val.batch(batch_size)

        ds_test = ds_provider.get_data("test")
        ds_test = prepare_ds('test',
                             ds_test,
                             add_noise = False,
                             add_empty = False,
                             augment = False,
                             hparams = hparams)
        ds_test = ds_test.batch(batch_size)

        # batch = ds_train.take(1)
        # data = batch.unbatch()
        # import numpy as np
        # import matplotlib.pyplot as plt
        # for inp, outp in data:
        #     fig = plt.figure()
        #     input = np.squeeze(inp)
        #     output = np.squeeze(outp)
        #     output = output.transpose([2, 0, 1])
        #     ax = fig.add_subplot(421)
        #     ax.imshow(input)
        #     i = 8
        #     stacked = input
        #     for pic in output:
        #         ax = fig.add_subplot(4, 7, i)
        #         ax.imshow(pic)
        #         stacked = stacked + pic
        #         i = i + 1
        #     ax = fig.add_subplot(422)
        #     ax.imshow(stacked)
        #     fig.show()
        # quit()

        logger.info("Making model...")
        model = make_model(hparams)

        try:
            train_pose_estimator(
                    model = model,
                    train_data = ds_train,
                    validation_data = ds_val,
                    test_data = ds_test,
                    hparams = hparams,
                    run_id = run_id,
                    tensorboard_dir = run_logdir,
                    checkpoint_dir = run_checkpoint_dir
                    )
        except Exception as ex:
            logger.exception(ex)

        del model  # delete the model to free memory

        # Try to copy the results over to the share as
        copy_success = False
        while not copy_success:
            try:
                # Copy results to remote
                logger.info("Copying tensorboard results for run {} to remote...".format(run_id))
                shutil.copytree(run_logdir, remote_tensorboard_dir)

                logger.info("Copying checkpoint results for run {} to remote...".format(run_id))
                shutil.copytree(run_checkpoint_dir, remote_checkpoint_dir)
            except Exception as ex:
                logger.exception(ex)
                time.sleep(10)
            else:
                copy_success = True


if __name__ == '__main__':

    with open("telegram_access.json", "r") as f:
        telegram_settings = json.load(f)
        telegram_user = telegram_settings['user']
        telegram_token = telegram_settings['token']
        telegram_chat = telegram_settings['chat']

    logdir_remote = "E:\\MasterDaten\\Results\\pose_est_2d\\uni\\remote"
    base_logdir = "E:\\MasterDaten\\Results\\pose_est_2d\\uni\\local"

    dataset_cache_path = "I:\\dataset_cache"

    selected_dataset = "NYU224ConfMap"

    batch_size = 17
    net_input_width = 224
    net_input_height = 224
    num_skel_joints = 21

    # Get pretrained encoder
    encoder_pretrained_gdrive_id = "1-v1_CJcT78D_OGu4y1ZFxs5g9Wfogi4_"
    encoder_pretrained_path = os.path.join(dataset_cache_path, "pretrained_models", "mobilenetv2_1.00_224).hdf5")
    gdd.download_file_from_google_drive(encoder_pretrained_gdrive_id,
                                        dest_path = encoder_pretrained_path,
                                        showsize = True,
                                        overwrite = False)

    # Get additional augmentation data
    augmentation_data_gdrive_id = "1y9NmvBWqYsaShRxhgfULLJ0dTfbr4JCk"
    augmentation_data_path = os.path.join(dataset_cache_path, "Augmentation_StructuralLearning")
    gdd.download_file_from_google_drive(augmentation_data_gdrive_id,
                                        dest_path = os.path.join(augmentation_data_path, "download.zip"),
                                        showsize = True,
                                        overwrite = False,
                                        unzip = True
                                        )
    empty_background_path = augmentation_data_path + "\\augmentation\\**\\*.png"

    with open("datasets.json", "r") as f:
        ds_settings = json.load(f)

    # Trials should test:
    HP_L1REGULARIZATION = hp.HParam('l1_regularization',
                                    display_name = 'L1Regularization',
                                    description = 'L1Regularization',
                                    domain = hp.Discrete([0, 1, 2]))

    HP_L2REGULARIZATION = hp.HParam('l2_regularization',
                                    display_name = 'L2Regularization',
                                    description = 'L2Regularization',
                                    domain = hp.Discrete([0, 1, 2]))

    HP_DROPOUT = hp.HParam('dropout',
                           display_name = 'Dropout Rate [%]',
                           description = 'Dropout Rate [%]',
                           domain = hp.Discrete([0, 10, 20]))

    HP_AUGMENTATION = hp.HParam('augmentation',
                                display_name = 'Augmentation Rate [%]',
                                description = 'Augmentation Rate [%]',
                                domain = hp.Discrete([25, 50]))

    HP_DECODERLAYERS = hp.HParam('decoderlayers',
                                 display_name = 'Number of layers in decoder',
                                 description = 'Number of layers in decoder',
                                 domain = hp.Discrete([4, 5]))

    HP_LOSS = hp.HParam("lossfunc", hp.Discrete(["skewed", "mse"]))

    HP_PRETRAINING = hp.HParam("pretraining", hp.Discrete([True, False]))

    HP_LEARNINGRATE = hp.HParam("learningrate",
                                display_name = 'Learning rate (x10**5)',
                                description = 'Learning rate (x10**5)',
                                domain = hp.Discrete([30, 50, 70, 100, 500]))

    HPARAMS = [
            HP_L1REGULARIZATION,
            HP_L2REGULARIZATION,
            HP_DROPOUT,
            HP_AUGMENTATION,
            HP_DECODERLAYERS,
            HP_LOSS,
            HP_PRETRAINING,
            HP_LEARNINGRATE,
            ]

    METRICS = [
            hp.Metric("epoch_accuracy", group = "validation", display_name = "accuracy (val.)", ),
            hp.Metric("epoch_loss", group = "validation", display_name = "loss (val.)", ),
            hp.Metric("batch_accuracy", group = "train", display_name = "accuracy (train)", ),
            hp.Metric("batch_loss", group = "train", display_name = "loss (train)", ),
            ]

    if not os.path.exists(logdir_remote):
        with tf.summary.create_file_writer(logdir_remote).as_default():
            hp.hparams_config(hparams = HPARAMS, metrics = METRICS)

    temp_dict = { domain: domain.domain.values for domain in HPARAMS }

    runs = [dict(zip(temp_dict, v)) for v in product(*temp_dict.values())]

    run(runs)
