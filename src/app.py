import argparse
import sys
import cv2

from tensorflow.python.platform import app

from models import hand_pose_estimator as hpe, gesture_classifier as gc
from tools.realsense import RealsenseCamera, RealsenseSettings
from tools import loggingutil


def main(argv):
    # TODO: load model for pose estimation
    pose_model = hpe.make_model()

    # TODO: load model for gesture recognition
    # gesture_model = gc.make_model()

    camera = None
    while camera is None:
        try:
            camera = RealsenseCamera()
        except Exception as e:
            logger.error("No Realsense camera found!")
            logger.error(e)
    presets = camera.get_depth_presets()
    camera.set_depth_preset(presets["High Accuracy"])

    # TODO: Segment hand from input image
    # TODO: estimate pose from segmented hand
    # TODO: classify gesture from stream of po0ses and react accordingly

    win_name = "my_window"
    cv2.namedWindow(win_name, cv2.WINDOW_FULLSCREEN)
    cv2.resizeWindow(win_name, camera.get_current_intrinsics().width, camera.get_current_intrinsics().height)

    for i in range(9999):
        last_frame_time, last_frame_depth, last_frame_rgb = camera.get_frame()

        # cv2.resizeWindow(win_name, rsc.get_current_intrinsics().width, rsc.get_current_intrinsics().height)
        cv2.imshow(win_name, last_frame_depth)
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
    logger = loggingutil.get_logger(__name__, do_file_logging=False)

    # if this file is called directly, start training of the rdf
    app.run(main=main, argv=[sys.argv[0]] + unparsed)
