import argparse
import sys
import cv2
import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use('Qt5Agg')

from tensorflow.python.platform import app

import tools
import models.hand_pose_estimator
import models.gesture_classifier

import tensorflow as tf
import numpy as np



last_frame_time = None
last_frame_depth = None
last_frame_rgb = None


def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        pass
    elif event == cv2.EVENT_LBUTTONUP:
        pass
    elif event == cv2.EVENT_MOUSEMOVE:
        logger.info(last_frame_depth[y][x])


input_record_width = 1280
input_record_height = 720
input_target_width = 128
input_target_height = 128

# pose estimation settings
# training:
train_pose_model = True
test_pose_model = True
dataset_source = 'NYU'

pose_train_data_dir = {
        'FHAD': r"E:\MasterDaten\Datasets\FHAD",
        'NYU': r"G:\master_thesis_data\Datasets\nyu\nyu_hand_dataset_v2\dataset"
        }
pose_train_data_dir = pose_train_data_dir[dataset_source]

if dataset_source == 'NYU':
    from tools import NYU as selected_dataset
elif dataset_source == 'FHAD':
    from tools import FHAD as selected_dataset

pose_batch_size = 128
pose_epochs = 400
pose_data_scale = 1 / 2 ** 16
pose_learning_rate = 0.0001


def overlay_skeleton(img, skel_cam_coord, skew_factor=None):
    skeleton_2d = tools.skeleton_renderer.project_2d(skel_cam_coord, selected_dataset.get_camera_intrinsics())
    if skew_factor:
        skeleton_2d = skeleton_2d.dot(np.array([[skew_factor[1], 0], [0, skew_factor[0]]]))
    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(img)
    image2 = cv2.convertScaleAbs(img, alpha=255 / maxVal)
    image2 = cv2.applyColorMap(image2, cv2.COLORMAP_PARULA)
    tools.render_skeleton(image2, skeleton_2d, joint_names=selected_dataset.get_joint_names())
    return image2


def prepare_dataset(ds):
    return ds.map(
            lambda img, skel: (tf.image.resize(img, tf.constant([input_target_height, input_target_width], dtype=tf.dtypes.int32)), skel),
            num_parallel_calls=tf.data.experimental.AUTOTUNE).map(
            lambda img, skel: (tf.cast(img, dtype=tf.float32) / tf.constant(5000, dtype=tf.float32),
                               (skel + tf.constant(2500.0, dtype=tf.float32)) / tf.constant(5000.0, dtype=tf.float32)))


def main(argv):
    pose_model = models.hand_pose_estimator.make_model(input_shape=(input_target_height, input_target_width, 1))
    logger.info("Hand pose model initialized!")

    if train_pose_model:
        logger.info("Hand pose training is active. Collecting dataset...")
        hand_pose_dataset = selected_dataset.get_dataset(pose_train_data_dir, 'train')
        hand_pose_dataset_val = selected_dataset.get_dataset(pose_train_data_dir, 'validation')

        logger.info("Preparing dataset...")
        hand_pose_dataset = prepare_dataset(hand_pose_dataset)
        hand_pose_dataset_val = prepare_dataset(hand_pose_dataset_val)

        # win_name = "my_window"
        # cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        # for img, skel in hand_pose_dataset.take(50000):
        #         #     img = img.numpy()
        #         #     skel = skel.numpy().reshape((21, -1))
        #         #     skel = skel * 5000.0 - 2500.0
        #         #     cv2.imshow(win_name, cv2.UMat(overlay_skeleton(img, skel, (input_target_height / 480, input_target_width / 640))))
        #         #     cv2.waitKey(1)
        #         # cv2.destroyAllWindows()

        hand_pose_dataset = hand_pose_dataset.shuffle(pose_batch_size * 10)

        hand_pose_dataset_val = hand_pose_dataset_val.batch(batch_size=pose_batch_size).prefetch(tf.data.experimental.AUTOTUNE)
        hand_pose_dataset = hand_pose_dataset.batch(batch_size=pose_batch_size).prefetch(tf.data.experimental.AUTOTUNE)

        logger.info("Starting training...")
        models.hand_pose_estimator.train_model(pose_model, hand_pose_dataset, pose_batch_size, pose_epochs, pose_learning_rate, hand_pose_dataset_val)

        # TODO: load model for gesture recognition
        # gesture_model = gc.make_model()

    if test_pose_model:
        logger.info("Hand pose test is active. Collecting dataset...")
        hand_pose_dataset_test = selected_dataset.get_dataset(pose_train_data_dir, 'test')

        logger.info("Preparing dataset...")
        hand_pose_dataset_test = prepare_dataset(hand_pose_dataset_test)

        win_name = "my_window"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

        for img, skel in hand_pose_dataset_test.take(50000):
            skeleton = models.hand_pose_estimator.estimate_pose(pose_model, img.numpy().reshape((-1, 128, 128, 1)))
            skeleton = skeleton.reshape((21, -1))
            skeleton = skeleton - 2500
            cv2.imshow(win_name, cv2.UMat(overlay_skeleton(img.numpy(), skeleton, (input_target_height / 480, input_target_width / 640))))
            cv2.waitKey(300)
        cv2.destroyAllWindows()

    camera = None
    while camera is None:
        try:
            camera = tools.RealsenseCamera()
        except Exception as e:
            logger.error("No Realsense camera found!")
            logger.error(e)
    presets = camera.get_depth_presets()
    camera.set_depth_preset(presets["High Accuracy"])

    win_name = "my_window"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, camera.get_current_intrinsics().width, camera.get_current_intrinsics().height)

    global last_frame_time
    global last_frame_depth
    global last_frame_rgb

    cv2.setMouseCallback(win_name, mouse_callback, None);

    for i in range(9999):
        last_frame_time, last_frame_depth, last_frame_rgb = camera.get_frame()

        last_frame_depth2 = cv2.resize(last_frame_depth, (input_target_height, input_target_width))
        model_input = last_frame_depth2.reshape((-1, 128, 128, 1))

        skeleton = models.hand_pose_estimator.estimate_pose(pose_model, model_input)
        skeleton = skeleton.reshape((21, -1))
        skeleton = skeleton - 2500

        # TODO: classify gesture from stream of poses and react accordingly

        cv2.imshow(win_name, cv2.UMat(overlay_skeleton(last_frame_depth2, skeleton, (input_target_height / 480, input_target_width / 640))))
        key = cv2.waitKey(1)


def train_pose_estimation():
    pass


def train_gesture_classification():
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
            '--model_dir',
            type=str,
            default='save',
            help='Base directory for output models.'
            )
    parser.add_argument(
            '--data_dir',
            type=str,
            default='tmp/data/',
            help='Directory for storing data'
            )
    parser.add_argument(
            '--train_steps',
            type=int,
            default=1000,
            help='Number of training steps.'
            )
    parser.add_argument(
            '--batch_size',
            type=str,
            default=1000,
            help='Number of examples in a training batch.'
            )
    FLAGS, unparsed = parser.parse_known_args()
    logger = tools.get_logger(__name__, do_file_logging=False)

    # if this file is called directly, start training of the rdf
    app.run(main=main, argv=[sys.argv[0]] + unparsed)
