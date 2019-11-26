import argparse
import sys
import cv2
import os
import matplotlib.pyplot as plt
import matplotlib
from datasets.tfrecord_helper import depth_and_skel

import json

from tensorflow.python.platform import app

import models.hand_pose_estimator
import models.gesture_classifier

import tensorflow as tf
import numpy as np

import tools
from datasets import SerializedDataset



configuration = None

last_frame_time = None
last_frame_depth = None
last_frame_rgb = None


def mouse_callback(x, y, img, event):
    if event == cv2.EVENT_LBUTTONDOWN:
        pass
    elif event == cv2.EVENT_LBUTTONUP:
        pass
    elif event == cv2.EVENT_MOUSEMOVE:
        logger.info(img[y][x])


def overlay_skeleton(img, skel_cam_coord, skew_factor=None, intrinsics=None, joint_names=None):
    skeleton_2d = tools.skeleton_renderer.project_2d(skel_cam_coord, intrinsics)
    if skew_factor:
        skeleton_2d = skeleton_2d.dot(np.array([[skew_factor[1], 0], [0, skew_factor[0]]]))
    image2 = tools.image_colorizer.colorize_cv(img, 0.0, 1.0, 'viridis')
    tools.render_skeleton(image2, skeleton_2d, joint_names=joint_names)
    return image2


def display_from_pose_dataset(ds, cam_intrinsics, joint_names=None, num_samples=1000, fps=30, width_factor=1.0, height_factor=1.0):
    win_name = "Displaying {} from dataset {} at {}fps".format(num_samples, ds, fps)
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    depth = None
    cv2.setMouseCallback(win_name, lambda event, x, y, flags, param: mouse_callback(x, y, depth, event))
    for img, skel in ds.take(num_samples):
        img = img.numpy()
        depth = img
        skel = skel.numpy().reshape((21, -1))
        cv2.imshow(win_name, cv2.UMat(overlay_skeleton(img, skel, (height_factor, width_factor), intrinsics=cam_intrinsics, joint_names=joint_names)))

        while cv2.waitKey(int(1000 / fps)) == 32:  # playback can be paused with space bar
            pass
    cv2.destroyAllWindows()


def prepare_dataset(ds, target_width, target_height):
    ds = ds.map(depth_and_skel, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.map(lambda img, skel: (
            tf.clip_by_value(tf.cast(tf.image.resize(img, tf.constant([target_height, target_width], dtype=tf.dtypes.int32)),
                                     dtype=tf.float32) / tf.constant(2500.0, dtype=tf.float32),
                             clip_value_min=0.0,
                             clip_value_max=1.0),
            skel), num_parallel_calls=tf.data.experimental.AUTOTUNE)  # ignore stuff more than 2.5m away.
    return ds


def train_pose_model(pose_est_conf, dataset_provider, model, data_dir, ds_img_width, ds_img_height):
    training_config = pose_est_conf['train']

    input_width = pose_est_conf['network_input_width']
    input_height = pose_est_conf['network_input_height']

    logger.info("Hand pose training is active. Collecting dataset...")

    logger.info("Preparing training dataset...")
    hand_pose_dataset = dataset_provider.get_data("train")
    hand_pose_dataset = prepare_dataset(hand_pose_dataset, target_width=input_width, target_height=input_height)

    if training_config['do_show_dataset_before_training']:
        display_from_pose_dataset(hand_pose_dataset,
                                  joint_names=dataset_provider.joint_names,
                                  width_factor=input_width / ds_img_width,
                                  height_factor=input_height / ds_img_height,
                                  cam_intrinsics=dataset_provider.camera_intrinsics)

    hand_pose_dataset = hand_pose_dataset.shuffle(training_config['batch_size'] * 10)
    hand_pose_dataset = hand_pose_dataset.batch(batch_size=training_config['batch_size']).prefetch(tf.data.experimental.AUTOTUNE)

    logger.info("Preparing validation dataset...")
    hand_pose_dataset_val = dataset_provider.get_data("validation")
    hand_pose_dataset_val = prepare_dataset(hand_pose_dataset_val, target_width=input_width, target_height=input_height)
    hand_pose_dataset_val = hand_pose_dataset_val.batch(batch_size=training_config['batch_size']).prefetch(tf.data.experimental.AUTOTUNE)

    logger.info("Starting training...")
    models.hand_pose_estimator.train_model(model=model,
                                           train_data=hand_pose_dataset,
                                           max_epochs=training_config['epochs'],
                                           learning_rate=training_config['learning_rate'],
                                           validation_data=hand_pose_dataset_val,
                                           data_dir=data_dir)


def test_pose_model(pose_est_conf, dataset_provider, model, ds_img_width, ds_img_height):
    model_input_width = pose_est_conf['network_input_width']
    model_input_height = pose_est_conf['network_input_height']

    logger.info("Preparing dataset...")
    hand_pose_dataset = dataset_provider.get_data("train")
    hand_pose_dataset = prepare_dataset(hand_pose_dataset, model_input_width, model_input_height)

    win_name = "my_window"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    for img, skel in hand_pose_dataset:
        skeleton = models.hand_pose_estimator.estimate_pose(model, img.numpy().reshape((-1, model_input_height, model_input_width, 1)))
        skeleton = skeleton.reshape((21, -1))
        cv2.imshow(win_name,
                   cv2.UMat(overlay_skeleton(img=img.numpy(),
                                             skel_cam_coord=skeleton,
                                             skew_factor=(model_input_height / ds_img_height, model_input_width / ds_img_width),
                                             intrinsics=dataset_provider.camera_intrinsics)))

        while cv2.waitKey(int(1000 / pose_est_conf['test_fps'])) == 32:
            pass
    cv2.destroyAllWindows()

    # TODO: Actually do test!


def main(argv):
    # POSE MODEL #
    input_image_width = configuration['input_image_width']
    input_image_height = configuration['input_image_height']
    output_data_dir = configuration['data_directory']

    pose_est_conf = configuration['pose_estimation']
    conf_ds = pose_est_conf['train']['use_dataset']
    pose_model_input_height = pose_est_conf['network_input_height']
    pose_model_input_width = pose_est_conf['network_input_width']

    pose_model = models.hand_pose_estimator.make_model(input_shape=(pose_model_input_height, pose_model_input_width, 1),
                                                       output_shape=pose_est_conf['num_network_outputs'],
                                                       data_dir=output_data_dir)
    logger.info("Hand pose model initialized!")

    ds_provider = SerializedDataset(ds_configuration[pose_est_conf["train"]["use_dataset"]])

    if pose_est_conf['do_training']:
        logger.info("Hand pose training is active.")
        train_pose_model(pose_est_conf=pose_est_conf,
                         dataset_provider=ds_provider,
                         model=pose_model,
                         data_dir=output_data_dir,
                         ds_img_width=input_image_width,
                         ds_img_height=input_image_height)

    if pose_est_conf['do_test']:
        logger.info("Hand pose test is active. ")
        test_pose_model(pose_est_conf=pose_est_conf,
                        dataset_provider=ds_provider,
                        model=pose_model,
                        ds_img_width=input_image_width,
                        ds_img_height=input_image_height)

    gesture_conf = configuration['gesture_classification']

    # TODO: load model for gesture recognition
    # gesture_model = gc.make_model()

    camera = None
    while camera is None:
        try:
            camera = tools.RealsenseCamera(settings=configuration['camera_settings'])
        except Exception as e:
            logger.error("No Realsense camera found!")
            logger.error(e)
    # presets = camera.get_depth_presets()
    # camera.set_depth_preset(presets["High Accuracy"])

    win_name = "my_window"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    global last_frame_time
    global last_frame_depth
    global last_frame_rgb

    last_frame_depth2 = None
    cv2.setMouseCallback(win_name, lambda event, x, y, flags, param: mouse_callback(x, y, last_frame_depth2, event), None)
    for i in range(9999):
        last_frame_time, last_frame_depth, last_frame_rgb = camera.get_frame()

        last_frame_depth2 = cv2.resize(last_frame_depth, (pose_model_input_height, pose_model_input_width))
        last_frame_depth2 = last_frame_depth2 / 2500.0
        last_frame_depth2 = np.clip(last_frame_depth2, 0.0, 1.0)

        model_input = last_frame_depth2.reshape((-1, pose_model_input_height, pose_model_input_width, 1))

        skeleton = models.hand_pose_estimator.estimate_pose(pose_model, model_input)
        skeleton = skeleton.reshape((21, -1))

        # TODO: classify gesture from stream of poses and react accordingly

        cv2.imshow(win_name, cv2.UMat(
                overlay_skeleton(last_frame_depth2,
                                 skeleton,
                                 skew_factor=(pose_model_input_height / input_image_height, pose_model_input_width / input_image_width),
                                 intrinsics=ds_provider.camera_intrinsics)))

        while cv2.waitKey(1) == 32:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
            '--config_file',
            type=str,
            default='config.json',
            help='Configuration file in json format.'
            )
    parser.add_argument(
            '--dataset_config',
            type=str,
            default='datasets.json',
            help='Configuration file in json format.'
            )
    FLAGS, unparsed = parser.parse_known_args()

    logger = tools.get_logger(__name__, do_file_logging=False)

    if FLAGS.config_file and not os.path.exists(FLAGS.config_file):
        logger.error("Config file {} does not exist!".format(FLAGS.config_file))
        exit(-1)

    with open(FLAGS.config_file, "r") as f:
        try:
            configuration = json.load(f)
        except Exception as e:
            logger.exception(e)
            exit(-1)

    with open(FLAGS.dataset_config, "r") as f:
        try:
            ds_configuration = json.load(f)
        except Exception as e:
            logger.exception(e)
            exit(-1)

    app.run(main=main, argv=[sys.argv[0]] + unparsed)
