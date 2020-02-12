import json
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import datasets.util
import tools
import tools.training_callbacks
from datasets import SerializedDataset
from datasets.tfrecord_helper import depth_and_skel
from models import models



logger = tools.get_logger('train_3d_pose_est', do_file_logging = True)

net_input_width = 224
net_input_height = 224
output_data_dir = "E:\\MasterDaten\\Results\\pose_est_3d_new"
batch_size = 50
max_epochs = 500
learning_rate = 0.005
tensorboard_dir = os.path.join(output_data_dir, 'tensorboard')
checkpoint_dir = os.path.join(output_data_dir, 'checkpoints')
final_save_name = os.path.join(checkpoint_dir, 'pose_est_final.hdf5')
refined_save_name = os.path.join(checkpoint_dir, 'pose_est_refined.hdf5')
checkpoint_prefix = 'cp_3d_pose_epoch'
num_skel_joints = 21
encoder_pretrained = "E:\\Google Drive\\UNI\\Master\\Thesis\\Data\\structural\\ae_with_bg_high_acc\\partial_save_46_layer_0(mobilenetv2_1.00_224).hdf5"


def getCrop(img, xstart, xend, ystart, yend, zstart, zend, thresh_z = True, background = 0):
    """
    Source: https://github.com/moberweger/deep-prior-pp/blob/master/src/util/handdetector.py#L260
    """
    if len(img.shape) == 2:
        cropped = img[max(ystart, 0):min(yend, img.shape[0]), max(xstart, 0):min(xend, img.shape[1])].copy()
        # add pixels that are out of the image in order to keep aspect ratio
        cropped = np.pad(cropped, ((abs(ystart) - max(ystart, 0),
                                    abs(yend) - min(yend, img.shape[0])),
                                   (abs(xstart) - max(xstart, 0),
                                    abs(xend) - min(xend, img.shape[1]))),
                         mode = 'constant', constant_values = background)
    elif len(img.shape) == 3:
        cropped = img[max(ystart, 0):min(yend, img.shape[0]), max(xstart, 0):min(xend, img.shape[1]), :].copy()
        # add pixels that are out of the image in order to keep aspect ratio
        cropped = np.pad(cropped, ((abs(ystart) - max(ystart, 0),
                                    abs(yend) - min(yend, img.shape[0])),
                                   (abs(xstart) - max(xstart, 0),
                                    abs(xend) - min(xend, img.shape[1])),
                                   (0, 0)),
                         mode = 'constant', constant_values = background)
    else:
        raise NotImplementedError()
    if thresh_z is True:
        msk1 = np.logical_and(cropped < zstart, cropped != 0)
        msk2 = np.logical_and(cropped > zend, cropped != 0)
        cropped[msk1] = zstart
        cropped[msk2] = 0.  # backface is at 0, it is set later
    return cropped


def extract_cube(depth_map, skeleton, cube_centroid, intr, crop_size = (224, 224)):
    CUBE_SIZE = 300  # mm
    HALF_CUBE_LENGTH = CUBE_SIZE / 2.0

    scale_augment = np.random.normal(1, 0.14) # 0.02 scale variance from depprior++, oberweger et al.
    HALF_CUBE_LENGTH = HALF_CUBE_LENGTH * scale_augment

    translate_augment = np.random.normal(0, 2.23)  # 5mm translation variance from depprior++, oberweger et al.
    skeleton = skeleton + translate_augment

    def np_extract_cube(depth_map, keypoint, intr):

        zstart = keypoint[2] - HALF_CUBE_LENGTH
        zend = keypoint[2] + HALF_CUBE_LENGTH
        xstart_3d = keypoint[0] - HALF_CUBE_LENGTH
        xend_3d = keypoint[0] + HALF_CUBE_LENGTH
        ystart_3d = keypoint[1] - HALF_CUBE_LENGTH
        yend_3d = keypoint[1] + HALF_CUBE_LENGTH

        cube_coords = np.array([
                [xstart_3d, ystart_3d, zstart],
                [xend_3d, ystart_3d, zstart],
                [xend_3d, yend_3d, zstart],
                [xstart_3d, yend_3d, zstart],

                [xstart_3d, ystart_3d, zend],
                [xend_3d, ystart_3d, zend],
                [xend_3d, yend_3d, zend],
                [xstart_3d, yend_3d, zend],
                ]
                )
        cube_2d = tools.project_2d(cube_coords, intr)
        cube_2d = np.round(cube_2d).astype(np.int32)
        start_2d = np.min(cube_2d, axis = 0)
        end_2d = np.max(cube_2d, axis = 0)

        crop = getCrop(depth_map, start_2d[0], end_2d[0], start_2d[1], end_2d[1], zstart, zend)

        # crop = crop - keypoint[2]  # normalize around keypoint
        # crop = crop * 1.0 / HALF_CUBE_LENGTH
        #
        # msk1 = np.logical_and(crop < -1.0, crop != 0)
        # msk2 = np.logical_and(crop > 1.0, crop != 0)
        # crop[msk1] = -1.0
        # crop[msk2] = 1.0

        max_side = np.max(crop.shape)
        z = np.zeros((max_side, max_side, 1))
        z[0:crop.shape[0], 0:crop.shape[1]] = crop

        crop = cv2.resize(z, (crop_size[0], crop_size[1]))
        crop = np.expand_dims(crop, 2)
        return crop.astype(np.float32)


    skeleton = tf.reshape(skeleton, [-1, 3])
    keypoint = skeleton[cube_centroid]
    result = tf.numpy_function(np_extract_cube, [depth_map, keypoint, intr],
                               Tout = tf.dtypes.float32,
                               name = "extract_cube")

    skeleton_cube = skeleton - keypoint
    skeleton_cube = skeleton_cube * 1 / HALF_CUBE_LENGTH
    skeleton_cube = tf.reshape(skeleton_cube, [-1])

    result = tf.ensure_shape(result, [crop_size[0], crop_size[1], 1], name = 'EnsureShapeAfterCubeExtract')
    return result, skeleton_cube


def prepare_ds(name, ds, cam_intr, add_noise, augment):
    CUBE_CENTROID = 3  # mmcp

    ds = ds.map(depth_and_skel, num_parallel_calls = tf.data.experimental.AUTOTUNE)

    ds = ds.map(lambda depth, skeleton: extract_cube(depth, skeleton, CUBE_CENTROID, cam_intr,
                                                     crop_size = (net_input_height, net_input_width)),
                num_parallel_calls = tf.data.experimental.AUTOTUNE)

    if add_noise:
        ds = ds.map(lambda img, skeleton:
                    (datasets.util.add_random_noise(img),
                     skeleton),
                    num_parallel_calls = tf.data.experimental.AUTOTUNE)
    return ds


def train_pose_estimator(train_data, validation_data, test_data,
                         saved_model = None, skip_pretraining = False):
    tools.clean_tensorboard_logs(tensorboard_dir)

    logger.info("Making model...")
    pose_estimator = models.make_model('3d-pose-est',
                                       input_shape = [net_input_height,
                                                      net_input_width, 1],
                                       num_joints = num_skel_joints,
                                       encoder_type = 'oberweger',
                                       encoder_weights = encoder_pretrained,
                                       regressor_weights = None
                                       )

    pose_estimator.build(input_shape = tf.TensorShape(
            [batch_size, net_input_height, net_input_width, 1]))
    pose_estimator.summary()

    telegram_callback = tools.training_callbacks.TelegramCallback(telegram_user, telegram_token, telegram_chat,
                                                                  "SingleTrainingRun")

    if not skip_pretraining:
        #######################################################################
        ##########################  Main training  ############################
        #######################################################################
        logger.info('Disabling training for encoder')
        pose_estimator.get_layer(index = 1).trainable = False
        pose_estimator.summary()

        if saved_model:
            pose_estimator.load_weights(saved_model)

        pose_estimator.get_layer(index = 1).trainable = True

        logger.info("Starting training...")
        models.train_model(
                model = pose_estimator,
                train_data = train_data,
                validation_data = validation_data,
                max_epochs = 20,
                learning_rate = learning_rate,
                tensorboard_dir = tensorboard_dir,
                checkpoint_dir = checkpoint_dir,
                best_cp_metric = 'acc',
                save_best_cp_only = True,
                cp_save_freq = 100000,
                cp_name = checkpoint_prefix,
                loss = tf.keras.losses.Huber(),
                verbose = 1,
                custom_callbacks = [telegram_callback],
                lr_reduce_patience = 2,
                lr_reduce_metric = 'loss',
                early_stop_patience = 5
                )

        logger.info("Preliminary Training done.")

        logger.info("Saving model as {}".format(final_save_name))
        pose_estimator.save(
                filepath = final_save_name,
                overwrite = True,
                include_optimizer = False,
                save_format = 'h5'
                )

        logger.info("Starting test...")
        test_result = pose_estimator.evaluate(
                x = test_data,
                verbose = 1
                )

        test_result_labeled = dict(
                zip(pose_estimator.metrics_names, test_result))
        print(test_result_labeled)

    ###########################################################################
    #########################  Refinement training    #########################
    ###########################################################################
    logger.info('Enabling training for encoder for fine tuning')
    pose_estimator.get_layer(index = 1).trainable = True

    if saved_model and skip_pretraining:
        # because of skip_pretraining we didn't already load the model in
        # the pretraining step
        try:
            pose_estimator.load_weights(saved_model)
        except:
            logger.info(
                    "Was not able to load saved weights {} with unlocked "
                    "encoder. Locking it temporarily.".format(
                            saved_model))
            pose_estimator.get_layer(index = 1).trainable = False
            pose_estimator.load_weights(saved_model)
            logger.info("Weights loaded. Unlocking encoder!")
            pose_estimator.get_layer(index = 1).trainable = True

    pose_estimator.summary()

    logger.info("Starting refinement training for encoder...")
    models.train_model(
            model = pose_estimator,
            train_data = train_data,
            validation_data = validation_data,
            max_epochs = max_epochs,
            learning_rate = learning_rate / 10,
            tensorboard_dir = tensorboard_dir,
            checkpoint_dir = checkpoint_dir,
            best_cp_metric = 'acc',
            save_best_cp_only = True,
            cp_save_freq = 100000,
            cp_name = checkpoint_prefix + "_refine_",
            loss = tf.keras.losses.Huber(),
            verbose = 1,
            custom_callbacks = [telegram_callback],
            lr_reduce_patience = 2,
            lr_reduce_metric = 'loss',
            early_stop_patience = 5
            )

    logger.info("Refinement Training done.")

    logger.info("Saving refined model as {}".format(refined_save_name))
    pose_estimator.save(
            filepath = refined_save_name,
            overwrite = True,
            include_optimizer = False,
            save_format = 'h5'
            )

    logger.info("Starting test...")
    test_result = pose_estimator.evaluate(
            x = test_data,
            verbose = 1
            )

    test_result_labeled = dict(zip(pose_estimator.metrics_names, test_result))
    logger.info(test_result_labeled)


if __name__ == '__main__':
    with open('telegram_access.json', 'r') as f:
        telegram_settings = json.load(f)
        telegram_user = telegram_settings['user']
        telegram_token = telegram_settings['token']
        telegram_chat = telegram_settings['chat']

    with open("datasets.json", "r") as f:
        ds_settings = json.load(f)

    ds_provider = SerializedDataset(ds_settings["BigHands"])

    ds_train = ds_provider.get_data("train")
    ds_train = prepare_ds('train',
                          ds_train,
                          cam_intr = ds_provider.camera_intrinsics,
                          add_noise = True,
                          augment = True)
    ds_train = datasets.util.batch_shuffle_prefetch(ds_train,
                                                    batch_size = batch_size)

    ds_val = ds_provider.get_data("validation")
    ds_val = prepare_ds('validation',
                        ds_val,
                        cam_intr = ds_provider.camera_intrinsics,
                        add_noise = False,
                        augment = False)
    ds_val = ds_val.batch(batch_size)

    ds_test = ds_provider.get_data("test")
    ds_test = prepare_ds('test',
                         ds_test,
                         cam_intr = ds_provider.camera_intrinsics,
                         add_noise = False,
                         augment = False)
    ds_test = ds_test.batch(batch_size)

    # batch = ds_train.take(1)
    # data = batch.unbatch()
    #
    # for inp, outp in data:
    #     fig = plt.figure()
    #     input = np.squeeze(inp)
    #     print(input.min())
    #     print(input.max())
    #     output = np.squeeze(outp)
    #     print(output.min())
    #     print(output.max())
    #     ax = fig.add_subplot(111)
    #     ax.imshow(input)
    #     fig.show()
    # quit()
    import glob
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "*.hdf5"))

    if len(checkpoint_files) > 0:
        latest_file = max(checkpoint_files, key = os.path.getctime)
        print("Trying to use checkpoint {}!".format(latest_file))
        train_pose_estimator(ds_train,
                             ds_val,
                             ds_test,
                             saved_model = latest_file,
                             skip_pretraining = ("refine" in latest_file)
                             )
    else:
        train_pose_estimator(ds_train,
                             ds_val,
                             ds_test)



