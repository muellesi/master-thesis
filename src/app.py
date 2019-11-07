import argparse
import sys
import cv2
import matplotlib.pyplot as plt

from tensorflow.python.platform import app

import tools
import models.hand_pose_estimator
import models.gesture_classifier

import tensorflow as tf



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
input_target_width = 64
input_target_height = 64

# pose estimation settings
# training:
train_pose_model = True
pose_train_data_dir = r"E:\MasterDaten\Datasets\FHAD"
pose_batch_size = 50
pose_epochs = 7
pose_data_scale = 1 / 2 ** 16


def main(argv):
    pose_model = models.hand_pose_estimator.make_model(input_shape=(input_target_width, input_target_height, 1))
    logger.info("Hand pose model initialized!")

    if train_pose_model:
        logger.info("Hand pose training is active. Collecting dataset...")
        hand_pose_dataset = tools.FHAD.get_dataset(pose_train_data_dir, (input_target_height, input_target_width))

        logger.info("Preparing dataset...")
        hand_pose_dataset = hand_pose_dataset.shuffle(1000).batch(batch_size=pose_batch_size).prefetch(tf.data.experimental.AUTOTUNE)

        # https://medium.com/coinmonks/beginners-guide-to-feeding-data-in-tensorflow-part2-5e2506d75429
        logger.info("Starting training...")
        models.hand_pose_estimator.train_model(pose_model, hand_pose_dataset, pose_batch_size, pose_epochs)

    # TODO: load model for gesture recognition
    # gesture_model = gc.make_model()

    camera = None
    while camera is None:
        try:
            camera = tools.RealsenseCamera()
        except Exception as e:
            logger.error("No Realsense camera found!")
            logger.error(e)
    presets = camera.get_depth_presets()
    camera.set_depth_preset(presets["High Accuracy"])

    # TODO: Segment hand from input image
    # segmentation_filter = tools.RdfHandSegmenter()
    # segmentation_filter.fit_tree()
    # segmentation_filter = tools.SimpleHandSegmenter()

    # TODO: estimate pose from segmented hand
    # TODO: classify gesture from stream of po0ses and react accordingly

    win_name = "my_window"
    cv2.namedWindow(win_name, cv2.WINDOW_FULLSCREEN)
    cv2.resizeWindow(win_name, camera.get_current_intrinsics().width, camera.get_current_intrinsics().height)

    global last_frame_time
    global last_frame_depth
    global last_frame_rgb

    cv2.setMouseCallback(win_name, mouse_callback, None);

    fig, ax = plt.subplots(nrows=1, ncols=1)
    fig.show()

    for i in range(9999):
        last_frame_time, last_frame_depth, last_frame_rgb = camera.get_frame()

        # depth_only_hand = segmentation_filter.remove_background(last_frame_depth)
        ax.imshow(last_frame_depth)
        # cv2.resizeWindow(win_name, rsc.get_current_intrinsics().width, rsc.get_current_intrinsics().height)
        cv2.imshow(win_name, cv2.UMat(last_frame_depth))
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
