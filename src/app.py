import argparse
import sys
import cv2
import os
import matplotlib.pyplot as plt
import matplotlib
from datasets.tfrecord_helper import depth_and_skel

import json

from tensorflow.python.platform import app

import models.models as models
from models.knn_gesture_classifier import KNNClassifier

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



def main(argv):
    # POSE MODEL #
    input_image_width  = configuration['input_image_width']
    input_image_height = configuration['input_image_height']
    output_data_dir    = configuration['data_directory']

    pose_est_conf           = configuration['pose_estimation']
    conf_ds                 = pose_est_conf['train']['use_dataset']
    pose_model_input_height = pose_est_conf['network_input_height']
    pose_model_input_width  = pose_est_conf['network_input_width']

    pose_model = models.make_model(input_shape  = (pose_model_input_height, pose_model_input_width, 1),
                                   output_shape = pose_est_conf['num_network_outputs'],
                                   data_dir     = output_data_dir)
                                   
    logger.info("Hand pose model initialized!")

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

        skeleton = models.direct_hand_pose_estimator.estimate_pose(pose_model, model_input)
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
