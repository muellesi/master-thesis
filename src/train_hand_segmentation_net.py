import json
import os

import tensorflow as tf

import datasets.util
import tools
import tools.training_callbacks
from datasets import SerializedDataset
from models import models



logger = tools.get_logger('train_2d_pose_est', do_file_logging = True)

net_input_width = 224
net_input_height = 224
output_data_dir = "E:\\MasterDaten\\Results\\hand_segmentation"
batch_size = 50
max_epochs = 500
learning_rate = 0.0005
tensorboard_dir = os.path.join(output_data_dir, 'tensorboard')
checkpoint_dir = os.path.join(output_data_dir, 'checkpoints')
final_save_name = os.path.join(output_data_dir, 'pose_est_final.hdf5')
refined_save_name = os.path.join(output_data_dir, 'pose_est_refined.hdf5')
checkpoint_prefix = 'cp_hand_seg_epoch'
encoder_pretrained = "E:\\Google Drive\\UNI\\Master\\Thesis\\Data\\structural\\ae_with_bg_high_acc\\partial_save_46_layer_0(mobilenetv2_1.00_224).hdf5"
empty_background_path = 'E:\\MasterDaten\\Datasets\\StructuralLearning' \
                        '\\augmentation\\**\\*.png'


@tf.function
def segmentationmap_from_skeleton(depth, skeleton, intr):
    CUBE_OFFSET = 50  # mm


    def inner_np(depth, skeleton, intr):
        skeleton = np.reshape(skeleton, [21, 3])
        max3d = np.max(skeleton, axis = 0)
        min3d = np.min(skeleton, axis = 0)

        max_z = max3d[2]
        min_z = min3d[2]

        cube = np.array(
                [
                        [min3d[0] - CUBE_OFFSET, min3d[1] - CUBE_OFFSET, min3d[2] - CUBE_OFFSET],  # front top left
                        [max3d[0] + CUBE_OFFSET, min3d[1] - CUBE_OFFSET, min3d[2] - CUBE_OFFSET],  # front top right
                        [max3d[0] + CUBE_OFFSET, max3d[1] + CUBE_OFFSET, min3d[2] - CUBE_OFFSET],  # front bottom right
                        [min3d[0] - CUBE_OFFSET, max3d[1] + CUBE_OFFSET, min3d[2] - CUBE_OFFSET],  # front bottom left

                        [min3d[0] - CUBE_OFFSET, min3d[1] - CUBE_OFFSET, max3d[2] + CUBE_OFFSET],  # back top left
                        [max3d[0] + CUBE_OFFSET, min3d[1] - CUBE_OFFSET, max3d[2] + CUBE_OFFSET],  # back top right
                        [max3d[0] + CUBE_OFFSET, max3d[1] + CUBE_OFFSET, max3d[2] + CUBE_OFFSET],  # back bottom right
                        [min3d[0] - CUBE_OFFSET, max3d[1] + CUBE_OFFSET, max3d[2] + CUBE_OFFSET],  # back bottom right
                        ]
                )
        cube_hom2d = intr.dot(cube.transpose()).transpose()
        cube2d = (cube_hom2d / cube_hom2d[:, 2:])[:, :2]

        max2d = tf.reduce_max(cube2d, axis = 0)
        min2d = tf.reduce_min(cube2d, axis = 0)

        maxx = int(min(net_input_width, max(0, max2d[0])))
        minx = int(min(net_input_width, max(0, min2d[0])))
        maxy = int(min(net_input_height, max(0, max2d[1])))
        miny = int(min(net_input_height, max(0, min2d[1])))

        print(maxx, minx, maxy, miny)

        mask = np.zeros_like(depth, dtype = np.float32)
        mask[miny:maxy, minx:maxx, :] = depth[miny:maxy, minx:maxx, :]

        maxz = max_z + CUBE_OFFSET
        minz = min_z - CUBE_OFFSET

        z_lim = np.logical_or(mask > maxz, mask < minz)
        thresh = np.logical_and(mask >= 1.0, mask != 0.0)
        mask[z_lim] = 0.0
        mask[thresh] = 1.0
        #print(mask.shape)
        return mask


    res = tf.numpy_function(inner_np, (depth, skeleton, intr), Tout = tf.float32)
    res = tf.ensure_shape(res, [int(net_input_height), int(net_input_width), 1])
    with tf.device('/GPU:0'):
        res = datasets.util.gaussian_smooth(res, 1.0)
    return res


def prepare_ds(name, ds, cam_intr, add_noise, add_empty, augment):
    ds = ds.map(lambda index, img, img_width, img_height, skeleton, conf_maps: (
            img, segmentationmap_from_skeleton(img, skeleton, cam_intr)))

    ds = ds.map(lambda img, mask:
                (datasets.util.scale_clip_image_data(img, 1.0 / 1500.0), mask),
                num_parallel_calls = tf.data.experimental.AUTOTUNE)

    if add_noise:
        ds = ds.map(lambda img, mask:
                    (datasets.util.add_random_noise(img),
                     mask),
                    num_parallel_calls = tf.data.experimental.AUTOTUNE)

    if add_empty:
        ds_empty_imgs = datasets.util.make_img_ds_from_glob(
                empty_background_path,
                width = net_input_width,
                height = net_input_height,
                value_scale = 1.0 / 1500.0,
                shuffle = True)
        ds_empty_imgs = ds_empty_imgs.map(lambda img: (
                img, tf.zeros(
                        [net_input_height, net_input_width, 1])))
        ds = ds.concatenate(ds_empty_imgs)

    return ds


    def make_model()


def train_pose_estimator(train_data, validation_data, test_data,
                         saved_model = None, skip_pretraining = False):
    tools.clean_tensorboard_logs(tensorboard_dir)

    logger.info("Making model...")
    pose_estimator = models.make_model('2d-pose-est',
                                       input_shape = [net_input_height,
                                                      net_input_width, 1],
                                       num_joints = num_skel_joints,
                                       encoder_weights = encoder_pretrained,
                                       regressor_weights = None
                                       )

    pose_estimator.build(input_shape = tf.TensorShape(
            [batch_size, net_input_height, net_input_width, 1]))
    pose_estimator.summary()

    visu = tools.training_callbacks.ConfMapOutputVisualization(
            log_dir = os.path.join(tensorboard_dir, 'plots',
                                   'ConfMapOutputVisualization'),
            feed_inputs_display = test_data,
            plot_every_x_batches = 2000,
            confmap_labels = ds_provider.joint_names
            )

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

        logger.info("Starting training...")
        models.train_model(
                model = pose_estimator,
                train_data = train_data,
                validation_data = validation_data,
                max_epochs = max_epochs,
                learning_rate = learning_rate,
                tensorboard_dir = tensorboard_dir,
                checkpoint_dir = checkpoint_dir,
                save_best_cp_only = True,
                cp_name = checkpoint_prefix,
                loss = make_skewed_mse(-0.5),
                verbose = 1,
                add_metrics = [keypoint_error_metric],
                custom_callbacks = [visu, telegram_callback],
                lr_reduce_patience = 2,
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
            learning_rate = learning_rate / 100,
            tensorboard_dir = tensorboard_dir,
            checkpoint_dir = checkpoint_dir,
            save_best_cp_only = True,
            cp_name = checkpoint_prefix + "_refine_",
            loss = make_skewed_mse(-0.5),
            verbose = 1,
            add_metrics = [keypoint_error_metric],
            custom_callbacks = [visu, telegram_callback]
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

    ds_provider = SerializedDataset(ds_settings["BigHands224ConfMap"])

    ds_train = ds_provider.get_data("train")
    ds_train = prepare_ds('train',
                          ds_train,
                          cam_intr = ds_provider.camera_intrinsics,
                          add_noise = True,
                          add_empty = True,
                          augment = True)
    ds_train = datasets.util.batch_shuffle_prefetch(ds_train,
                                                    batch_size = batch_size)

    ds_val = ds_provider.get_data("validation")
    ds_val = prepare_ds('validation',
                        ds_val,
                        cam_intr = ds_provider.camera_intrinsics,
                        add_noise = False,
                        add_empty = False,
                        augment = False)
    ds_val = ds_val.batch(batch_size)

    ds_test = ds_provider.get_data("test")
    ds_test = prepare_ds('test',
                         ds_test,
                         cam_intr = ds_provider.camera_intrinsics,
                         add_noise = False,
                         add_empty = False,
                         augment = False)
    ds_test = ds_test.batch(batch_size)

    batch = ds_train.take(1)
    data = batch.unbatch()
    import numpy as np
    import matplotlib.pyplot as plt



    for inp, outp in data:
        fig = plt.figure()
        input = np.squeeze(inp)
        output = np.squeeze(outp)
        ax = fig.add_subplot(121)
        ax.imshow(input)
        ax = fig.add_subplot(122)
        im = ax.imshow(output)
        fig.colorbar(im)
        fig.show()
    quit()

    train_pose_estimator(ds_train,
                         ds_val,
                         ds_test,
                         skip_pretraining = True,
                         saved_model = "E:\\Google Drive\\UNI\\Master\\Thesis\\Data\\pose_est\\2d\\ueber_weihnachten\\pose_est_refined.hdf5"
                         )
