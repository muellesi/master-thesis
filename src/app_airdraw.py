import collections
import os
from datetime import datetime
from datetime import timedelta

import cv2
import numpy as np
import tensorflow as tf

import app_framework.actions
import app_framework.gui.main_window
import datasets.util
import refiners
import tools
from app_framework.gesture_save_file import deserialize_to_gesture_collection
from app_framework.gesture_save_file import serialize_gesture_collection
from models.knn_gesture_classifier import KNNClassifier
from tools import RealsenseCamera



logger = tools.get_logger("MainApp_Airdraw")
configuration = None

last_frame_time = None
last_frame_depth = None
last_frame_rgb = None

# pose_model_path = 'E:\\Google Drive\\UNI\\Master\\Thesis\\Data\\pose_est\\2d\\weiter_trainiert\\checkpoints\\cp_2d_pose_epoch_refine_.43.hdf5'
pose_model_path = 'E:\\Google Drive\\UNI\\Master\\Thesis\\Data\\pose_est\\2d\\noch_weiter_trainiert\\checkpoints\\pose_est_refined.hdf5'
camera_settings_file = 'E:\\Google Drive\\UNI\\Master\\Thesis\\ThesisCode\\src\\realsense_settings.json'
gesture_sample_length = 60
hand_detection_limit = 0.6

filter_param_mincutoff = 1.0
filter_param_beta = 0.01
filter_param_freq = 30

use_colormap = 'bone'


def twod_argmax(val):
    maxy = tf.argmax(tf.reduce_max(val, axis = 2), 1)
    maxx = tf.argmax(tf.reduce_max(val, axis = 1), 1)
    maxs = tf.stack([maxy, maxx], axis = 2)
    maxs = tf.cast(maxs, dtype = tf.dtypes.float32)
    return maxs


def get_closest_nonzero(depthmap, non_zero_thresh, erode_kernel = None):
    if erode_kernel is None:
        erode_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    non_zero_depth = np.where(np.greater(depthmap, non_zero_thresh), np.ones_like(depthmap),
                              np.zeros_like(depthmap))
    non_zero_mask = cv2.erode(non_zero_depth, erode_kernel, iterations = 1)

    non_zero_mask = np.where(np.greater(non_zero_mask, 0))
    closest_distance = np.min(depthmap[non_zero_mask])
    return closest_distance


def mask_depth_farther_than(depthmap, thresh, open_kernel = None):
    if open_kernel is None:
        open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    upper_mask = np.where(np.less_equal(depthmap, thresh + 200.0), np.ones_like(depthmap),
                          np.zeros_like(depthmap))
    upper_mask = cv2.medianBlur(upper_mask, 3)

    upper_mask = cv2.morphologyEx(upper_mask, cv2.MORPH_OPEN, open_kernel)

    mask_result = depthmap * upper_mask
    return mask_result


class AirdrawPoseChecker:

    def __init__(self):
        self.airdraw_active = False
        self.airdraw_disable_cooldown = 2
        self.airdraw_enable_cooldown = 2
        self.airdraw_enable_counter = 0
        self.airdraw_disable_counter = 0


    def check_airdraw_pose(self, coords):
        # check for gesture pose
        w_idx = 0
        tip_idxs = [8, 11, 14, 17, 20]
        w_dists = np.linalg.norm(coords - coords[w_idx], axis = 1)  # distance from all joints to the wrist
        w_tip_dists = w_dists[tip_idxs]  # for ease of use, preselect finger tip distances
        tip_mcp_dists = np.abs([w_tip_dists[i] - w_dists[i + 1] for i in range(5)])

        gesture_cond = True
        # gesture_cond = np.mean(w_tip_dists[[0, 1]]) > 2 * np.median(w_tip_dists[[0, 1, 2, 3, 4]], axis = 0)
        # gesture_cond = gesture_cond and w_tip_dists[0] > 1.2 * w_dists[7]
        gesture_cond = gesture_cond and tip_mcp_dists[1] > np.mean(tip_mcp_dists[[0, 2, 3, 4]])
        gesture_cond = gesture_cond and not np.allclose(coords, 0.0)  # zeropose is not valid

        if gesture_cond:
            if not self.airdraw_active:
                self.airdraw_enable_counter += 1
            if not self.airdraw_active and self.airdraw_enable_counter > self.airdraw_enable_cooldown:
                self.airdraw_active = True
                self.airdraw_enable_counter = 0
        else:
            if self.airdraw_active:
                self.airdraw_disable_counter += 1
            if self.airdraw_active and self.airdraw_disable_counter > self.airdraw_disable_cooldown:
                self.airdraw_active = False
                self.airdraw_disable_counter = 0
        return self.airdraw_active


def cv_from_frame(frame, model):
    closest_distance = get_closest_nonzero(frame, 105)
    depth = cv2.resize(frame, (224, 224))
    depth = mask_depth_farther_than(depth, closest_distance)

    depth_clipped = datasets.util.scale_clip_image_data(depth, 1.0 / 1500.0)

    # Pose estimation
    depth_clipped = np.expand_dims(np.expand_dims(depth_clipped, 2), 0)
    res = model.predict(depth_clipped)

    # Get Maximum coordinates
    coords = twod_argmax(res)
    coords = coords.numpy().squeeze()

    # Get Maximum values
    res = res.squeeze()
    values = tf.reduce_max(res, axis = [0, 1]).numpy()

    return coords, values


def airdraw_from_points(img, pts):
    if len(pts) > 2:
        pts = np.array(pts, dtype = np.int32)
        pts = np.flip(pts, 1)  # switch x and y coordinates for cv2's point format
        pts = np.expand_dims(pts, 0)  # polylines wants a 3d array

        cv2.polylines(img, pts, isClosed = False, color = (255, 0, 0), thickness = 5)
    return img


def record_sample(pose_estimation_model):
    display_size = (640, 480)
    countdown_font_scale = 5
    countdown_font_thickness = 2
    logger.info("recording sample...")
    cam = RealsenseCamera({ 'file': camera_settings_file })

    pose_checker = AirdrawPoseChecker()

    # Open Window
    win_name = "Sample record..."
    cv2.namedWindow(win_name)

    # Show Countdown
    ####################################################################
    countdown_start_time = datetime.now()
    countdown_end_time = countdown_start_time + timedelta(seconds = 5)
    while datetime.now() < countdown_end_time:
        time, depth_raw, rgb = cam.get_frame()  # camera warm up

        cntdwn_img = np.zeros((display_size[1], display_size[0], 3))

        display_time = countdown_end_time - datetime.now()
        if datetime.now() < countdown_end_time:
            font = cv2.FONT_HERSHEY_SIMPLEX
            (text_width, text_height) = \
                cv2.getTextSize(str(display_time.seconds), font, fontScale = countdown_font_scale,
                                thickness = countdown_font_thickness)[0]

            text_offset_x = int(display_size[0] / 2 - text_width / 2)  # Center horzontally
            text_offset_y = int(display_size[1] / 2 + text_height / 2)  # Center vertically

            cv2.putText(cntdwn_img, str(display_time.seconds), (text_offset_x, text_offset_y), font,
                        fontScale = countdown_font_scale, color = (255, 255, 255), thickness = countdown_font_thickness)

            cv2.imshow(win_name, cntdwn_img)
            cv2.waitKey(1)
    ##########################################################################
    # Countdown end

    sample_frames = collections.deque()
    one_euro = refiners.OneEuroFilter(freq = filter_param_freq, mincutoff = filter_param_mincutoff,
                                      beta = filter_param_beta)
    record_start_time = datetime.now()
    sample_frames_remaining = gesture_sample_length
    is_airdraw_active = False

    while sample_frames_remaining > 0:

        # Get Frame
        time, depth_raw, rgb = cam.get_frame()  # camera warm up

        coords, values = cv_from_frame(depth_raw, pose_estimation_model)
        mean_keypoint_peak_value = np.mean(values)

        if mean_keypoint_peak_value < hand_detection_limit:
            coords = np.zeros(coords.shape)

        coords = one_euro(coords, (datetime.now() - record_start_time).total_seconds())

        # prepare depth image for display
        if depth_raw.shape != (480, 640):
            depth_raw = cv2.resize(depth_raw, (640, 480))
        depth_raw = np.flip(depth_raw, 1)  # feels more natural
        prod_img = tools.colorize_cv(depth_raw.squeeze(), cmap = use_colormap)

        # Display hand skeleton
        if mean_keypoint_peak_value > hand_detection_limit:
            coords_display = np.copy(coords)
            coords_display[:, 1] = 224.0 - coords_display[:, 1]
            coords_scaled = coords_display * np.array([480 / 224, 640 / 224])
            tools.render_skeleton(prod_img, np.stack([coords_scaled[:, 1], coords_scaled[:, 0]], axis = 1), True,
                                  np.round(values, 3))

        old_is_airdraw_active = is_airdraw_active

        is_airdraw_active = pose_checker.check_airdraw_pose(coords)

        if is_airdraw_active:
            sample_frames.appendleft(coords[11])  # only use itip
            sample_frames_remaining -= 1

            cv2.putText(prod_img, "Samples remaining: {}".format(sample_frames_remaining), (320, 30),
                        cv2.FONT_HERSHEY_PLAIN, fontScale = 0.75,
                        color = (0, 0, 0), thickness = 2)
            cv2.putText(prod_img, "Samples remaining: {}".format(sample_frames_remaining), (320, 30),
                        cv2.FONT_HERSHEY_PLAIN, fontScale = 0.75,
                        color = (255, 255, 255), thickness = 1)

            sample_frames_scaled = np.stack(sample_frames)
            sample_frames_scaled[:, 1] = 224.0 - sample_frames_scaled[:, 1]
            sample_frames_scaled = sample_frames_scaled * np.array([480 / 224, 640 / 224])
            prod_img = airdraw_from_points(prod_img, sample_frames_scaled)

        elif not is_airdraw_active and old_is_airdraw_active != is_airdraw_active:
            break

        cv2.imshow(win_name, prod_img)
        cv2.waitKey(1)

    cv2.destroyAllWindows()
    del cam

    while len(sample_frames) < gesture_sample_length:
        sample_frames.appendleft(np.zeros((2)))
    return np.stack(sample_frames)


def element_diff(samples):
    if len(samples.shape) == 3:
        return np.diff(samples, axis = 0, append = 0)
    elif len(samples.shape) == 2:
        return np.diff(samples, axis = 1, append = 0)
    else:
        raise NotImplementedError()


def run_app(model, action_manager, gesture_data):
    pose_checker = AirdrawPoseChecker()
    is_airdraw_active = False

    cam = RealsenseCamera({ 'file': camera_settings_file })

    time, depth_raw, rgb = cam.get_frame()  # camera warm up
    coords, vals = cv_from_frame(depth_raw, model)  # model warm up

    gesture_classifier = KNNClassifier(k = 3,
                                       batch_size = gesture_sample_length,
                                       sample_shape = coords.size,
                                       metric = 'manhattan'
                                       )
    X = []
    Y = []

    for idx, gesture in enumerate(gesture_data):
        for sample in gesture.samples:
            X.append(sample.reshape(-1))
            Y.append(idx)

            # X.append(element_diff(sample).reshape(-1))
            # Y.append(idx + 1)

    gesture_classifier.set_train_data(X, Y)

    do_run = True
    display_results = True
    last_action_time = datetime.now()
    last_gesture = None
    last_coords = np.zeros(coords.shape)

    win_name = 'Self learning gesture control for automotive application...'
    cv2.namedWindow(win_name)

    one_euro = refiners.OneEuroFilter(freq = filter_param_freq, mincutoff = filter_param_mincutoff,
                                      beta = filter_param_beta)
    global_start_time = datetime.now()

    is_airdraw_active = False
    sample_frames_remaining = gesture_sample_length

    point_samples = collections.deque(maxlen = gesture_sample_length)

    while do_run:

        loop_start_time = datetime.now()
        time, depth_raw, rgb = cam.get_frame()  # camera warm up

        coords, vals = cv_from_frame(depth_raw, model)
        value_mean = np.mean(vals)

        if depth_raw.shape != (480, 640):
            depth_raw = cv2.resize(depth_raw, (640, 480))
        depth_raw = np.flip(depth_raw, 1)  # feels more natural
        result_img = tools.colorize_cv(depth_raw.squeeze(), cmap = use_colormap)

        if value_mean < hand_detection_limit:
            coords = np.zeros(coords.shape)
            skeleton_valid = False
        else:
            skeleton_valid = True

        coords = one_euro(coords, (datetime.now() - global_start_time).total_seconds())

        old_is_airdraw_active = is_airdraw_active
        is_airdraw_active = pose_checker.check_airdraw_pose(coords)

        if is_airdraw_active:
            point_samples.appendleft(coords[11])
            sample_frames_remaining -= 1

            sample_frames_scaled = np.stack(point_samples)
            sample_frames_scaled[:, 1] = 224.0 - sample_frames_scaled[:, 1]
            sample_frames_scaled = sample_frames_scaled * np.array([480 / 224, 640 / 224])
            airdraw_from_points(result_img, sample_frames_scaled)

        if (is_airdraw_active != old_is_airdraw_active and not is_airdraw_active) \
                or (old_is_airdraw_active and sample_frames_remaining == 0):
            if not is_airdraw_active and old_is_airdraw_active:
                print("Predicting because airdraw mode ended!")
            elif sample_frames_remaining == 0:
                print("Predicting because frame buffer is full!")

            if len(point_samples) > 10:  # only predict if we have enough frames
                while len(point_samples) < gesture_sample_length:
                    point_samples.appendleft(np.zeros(2))

                gesture_classifier.push_samples(point_samples)
                gesture_prediction = gesture_classifier.predict()[0]

                gesture = gesture_data[gesture_prediction]
                last_gesture = gesture
                action_manager.exec_action(gesture.action)

                # Current Gesture Probabilities
                class_probabilities = list(gesture_classifier.predict_proba())[0][1:]
                class_probabilities = dict(
                        zip([gesture_data[i].name for i in gesture_classifier.class_names], class_probabilities[0]))
                logger.info(class_probabilities)

            # reset everything
            sample_frames_remaining = gesture_sample_length
            gesture_classifier.reset_queue()
            point_samples.clear()

        end_time = datetime.now()

        coords_display = np.copy(coords)
        coords_display[:, 1] = 224.0 - coords_display[:, 1]

        # Skeleton
        if value_mean > hand_detection_limit:
            coords_scaled = coords_display * np.array([480 / 224, 640 / 224])
            tools.render_skeleton(result_img, np.stack([coords_scaled[:, 1], coords_scaled[:, 0]], axis = 1), True,
                                  np.round(vals, 2))

        # FPS counter
        fps = (1 / (end_time - loop_start_time).microseconds) * 1e6
        fps_text = "{:.01f} fps".format(fps)
        cv2.putText(result_img, fps_text, (10, 30), cv2.FONT_HERSHEY_PLAIN, fontScale = 0.75, color = (0, 0, 0),
                    thickness = 2)
        cv2.putText(result_img, fps_text, (10, 30), cv2.FONT_HERSHEY_PLAIN, fontScale = 0.75,
                    color = (255, 255, 255), thickness = 1)

        # Current Gesture Class
        last_name = "None" if last_gesture is None else last_gesture.name
        gesture_text = "Last gesture: {} ".format(last_name)
        cv2.putText(result_img, gesture_text, (320, 30), cv2.FONT_HERSHEY_PLAIN, fontScale = 0.75,
                    color = (0, 0, 0), thickness = 2)
        cv2.putText(result_img, gesture_text, (320, 30), cv2.FONT_HERSHEY_PLAIN, fontScale = 0.75,
                    color = (255, 255, 255), thickness = 1)

        cv2.imshow(win_name, result_img)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # esc
            do_run = False

    del cam
    cv2.destroyAllWindows()


def main(argv):
    gesture_data = []
    if os.path.exists(save_file_name):
        gesture_data = deserialize_to_gesture_collection(save_file_name)

    if not os.path.exists(pose_model_path):
        logger.error("Pose estimation model could not be found at {}".format(pose_model_path))
        return

    model = tf.keras.models.load_model(pose_model_path, compile = False)
    model.predict(np.zeros((1, 224, 224, 1)))  # warm up to prevent lag later

    action_manager = app_framework.actions.ActionManager()

    control_center = app_framework.gui.MainWindow(action_manager,
                                                  sample_record_callback = lambda: record_sample(model),
                                                  save_gestures_callback = lambda
                                                      gestures: serialize_gesture_collection(gestures, save_file_name),
                                                  main_app_callback = lambda samples: run_app(model, action_manager,
                                                                                              samples))
    control_center.set_gestures(gesture_data)
    while control_center.alive:
        try:
            control_center.update()
        except Exception as e:
            logger.exception(e)
            del control_center
            del model
            return

    del control_center
    del model
    return (0)


if __name__ == "__main__":
    save_file_name = "gesture_data_airdraw.json"
    main(None)
    del tf
    logger.info("App end!")

    from sys import exit



    exit(0)
