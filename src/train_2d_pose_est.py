import json
import os

import tensorflow as tf

import datasets.util
import tools
from datasets import SerializedDataset
from datasets.tfrecord_helper import depth_and_skel
from models import models



logger = tools.get_logger(__name__, do_file_logging = False)

net_input_width = 224
net_input_height = 224
output_data_dir = "E:\\MasterDaten\\Results\\pose_est_2d"
batch_size = 50
max_epochs = 50
learning_rate = 0.005
tensorboard_dir = os.path.join(output_data_dir, 'tensorboard')
checkpoint_dir = os.path.join(output_data_dir, 'checkpoints')
final_save_name = os.path.join(output_data_dir, 'pose_est_final.hdf5')
refined_save_name = os.path.join(output_data_dir, 'pose_est_refined.hdf5')
checkpoint_prefix = 'cp_'
num_skel_joints = 21
encoder_pretrained = "E:\\Google " \
                     "Drive\\UNI\\Master\\Thesis\\src\\data\\structural" \
                     "\\ae_with_bg_high_acc\\partial_save_46_layer_0(" \
                     "mobilenetv2_1.00_224).hdf5"
empty_background_path = 'E:\\MasterDaten\\Datasets\\StructuralLearning' \
                        '\\augmentation\\**\\*.png'


def prepare_ds(ds, add_noise, add_empty, cam_intr):
    ds = ds.map(depth_and_skel,
                num_parallel_calls = tf.data.experimental.AUTOTUNE)

    ds = ds.map(lambda img, skel:
                (img,
                 tf.numpy_function(
                         datasets.util.skel_to_confmaps,
                         [skel,
                          cam_intr,
                          net_input_width,
                          net_input_height,
                          net_input_width / ds_provider.depth_width,
                          net_input_height / ds_provider.depth_height,
                          10],
                         Tout = tf.float32)),
                num_parallel_calls = tf.data.experimental.AUTOTUNE)

    ds = ds.map(lambda img, skel:
                (img, tf.ensure_shape(skel, [224, 224, num_skel_joints])),
                num_parallel_calls = tf.data.experimental.AUTOTUNE)

    ds = ds.map(lambda img, skel:
                (datasets.util.scale_image(img, [net_input_height,
                                                 net_input_width]),
                 skel), num_parallel_calls = tf.data.experimental.AUTOTUNE)

    ds = ds.map(lambda img, skel:
                (datasets.util.scale_clip_image_data(img, 1.0 / 2500.0),
                 skel), num_parallel_calls = tf.data.experimental.AUTOTUNE)

    if add_noise:
        ds = ds.map(lambda img, skel:
                    (datasets.util.add_random_noise(img),
                     skel), num_parallel_calls = tf.data.experimental.AUTOTUNE)

    if add_empty:
        ds_empty_imgs = datasets.util.make_img_ds_from_glob(
                empty_background_path,
                width = net_input_width,
                height = net_input_height,
                value_scale = 1.0 / 2500.0,
                shuffle = True)
        ds_empty_imgs = ds_empty_imgs.map(lambda img: (
        img, tf.zeros([net_input_height, net_input_width, num_skel_joints])))

    return ds


def train_pose_estimator(train_data, validation_data, test_data,
                         saved_model = None):
    logger.info("Making model...")
    pose_estimator = models.make_model('2d-pose-est',
                                       input_shape = [net_input_height,
                                                      net_input_width, 1],
                                       num_joints = num_skel_joints,
                                       encoder_weights = encoder_pretrained,
                                       regressor_weights = None
                                       )

    if saved_model:
        pose_estimator.load_weights(saved_model)

    ###########################################################################
    ###########################  Main training    #############################
    ###########################################################################
    pose_estimator.set_encoder_trainable(False)

    logger.info("Starting training...")
    models.train_model(
            model = pose_estimator,
            train_data = train_data,
            validation_data = validation_data,
            max_epochs = max_epochs,
            learning_rate = learning_rate,
            tensorboard_dir = tensorboard_dir,
            do_clean_tensorboard_dir = True,
            checkpoint_dir = checkpoint_dir,
            save_best_cp_only = True,
            best_cp_metric = 'val_acc',
            cp_name = checkpoint_prefix,
            loss = tf.keras.losses.mean_squared_error
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
            batch_size = batch_size,
            verbose = 1
            )

    test_result_labeled = dict(zip(pose_estimator.metrics_names, test_result))
    print(test_result_labeled)

    ###########################################################################
    #########################  Refinement training    #########################
    ###########################################################################
    pose_estimator.set_encoder_trainable(True)

    logger.info("Starting refinement training for encoder...")
    models.train_model(
            model = pose_estimator,
            train_data = train_data,
            validation_data = validation_data,
            max_epochs = 10,
            learning_rate = learning_rate / 100,
            tensorboard_dir = tensorboard_dir,
            do_clean_tensorboard_dir = False,
            checkpoint_dir = checkpoint_dir,
            save_best_cp_only = True,
            best_cp_metric = 'val_acc',
            cp_name = checkpoint_prefix + "_refine_",
            loss = tf.keras.losses.mean_squared_error
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
            batch_size = batch_size,
            verbose = 1
            )

    test_result_labeled = dict(zip(pose_estimator.metrics_names, test_result))
    print(test_result_labeled)


if __name__ == '__main__':
    with open("datasets.json", "r") as f:
        ds_settings = json.load(f)

    ds_provider = SerializedDataset(ds_settings["BigHands"])

    ds_train = ds_provider.get_data("train")
    ds_train = prepare_ds(ds_train,
                          add_noise = True,
                          add_empty = True,
                          cam_intr = ds_provider.camera_intrinsics)
    ds_train = datasets.util.batch_shuffle_prefetch(ds_train,
                                                    batch_size = batch_size)

    ds_val = ds_provider.get_data("validation")
    ds_val = prepare_ds(ds_val,
                        add_noise = True,
                        add_empty = False,
                        cam_intr = ds_provider.camera_intrinsics)
    ds_val = datasets.util.batch_shuffle_prefetch(ds_val,
                                                  batch_size = batch_size)

    ds_test = ds_provider.get_data("test")
    ds_test = prepare_ds(ds_test,
                         add_noise = False,
                         add_empty = False,
                         cam_intr = ds_provider.camera_intrinsics)
    ds_val = datasets.util.batch_shuffle_prefetch(ds_val,
                                                  batch_size = batch_size)

    # batch = ds_train.take(1)
    # data = batch.unbatch()
    #
    # for inp, outp in data:
    #     fig = plt.figure(figsize = (20, 20))
    #     input = np.squeeze(inp)
    #     output = np.squeeze(outp)
    #     ax = fig.add_subplot(421)
    #     ax.imshow(input)
    #     i = 8
    #     stacked = input
    #     for pic in output:
    #         ax = fig.add_subplot(4, 7, i)
    #         ax.imshow(pic)
    #         stacked = stacked + pic * 1000
    #         i = i + 1
    #     ax = fig.add_subplot(422)
    #     ax.imshow(stacked)
    #     fig.show()

    train_pose_estimator(ds_train, ds_val, ds_test, saved_model = None)
