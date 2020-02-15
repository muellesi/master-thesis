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



logger = tools.get_logger("MainApp")
configuration = None

last_frame_time = None
last_frame_depth = None
last_frame_rgb = None

#pose_model_path = 'E:\\Google Drive\\UNI\\Master\\Thesis\\Data\\pose_est\\2d\\weiter_trainiert\\checkpoints\\cp_2d_pose_epoch_refine_.43.hdf5'
pose_model_path = 'E:\\Google Drive\\UNI\\Master\\Thesis\\Data\\pose_est\\2d\\noch_weiter_trainiert\\checkpoints\\pose_est_refined.hdf5'
camera_settings_file = 'E:\\Google Drive\\UNI\\Master\\Thesis\\ThesisCode\\src\\realsense_settings.json'
gesture_sample_length = 90
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


def record_sample(model):
    display_size = (640, 480)
    countdown_font_scale = 5
    countdown_font_thickness = 2
    print("recording sample...")
    cam = RealsenseCamera({
            'file': camera_settings_file })

    # Open Window
    win_name = "Sample record..."
    cv2.namedWindow(win_name)

    # Show Countdown

    start_time = datetime.now()
    end_time = start_time + timedelta(seconds = 5)
    while datetime.now() < end_time:
        time, depth_raw, rgb = cam.get_frame()  # camera warm up

        cntdwn_img = np.zeros((display_size[1], display_size[0], 3))

        display_time = end_time - datetime.now()
        if datetime.now() < end_time:
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

    sample_frames = collections.deque()

    one_euro = refiners.OneEuroFilter(freq = filter_param_freq, mincutoff = filter_param_mincutoff, beta = filter_param_beta)
    global_start_time = datetime.now()
    for i in range(gesture_sample_length):
        # Get Frame
        time, depth_raw, rgb = cam.get_frame()  # camera warm up
        coords, values = cv_from_frame(depth_raw, model)

        value_mean = np.mean(values)

        if value_mean < hand_detection_limit:
            coords = np.zeros(coords.shape)

        coords = one_euro(coords, (datetime.now() - global_start_time).total_seconds())
        sample_frames.appendleft(coords)

        if depth_raw.shape != (480, 640):
            depth_raw = cv2.resize(depth_raw, (640, 480))

        depth_raw = np.flip(depth_raw, 1)  # feels more natural
        coords_display = np.copy(coords)
        coords_display[:, 1] = 224.0 - coords_display[:, 1]

        prod_img = tools.colorize_cv(depth_raw.squeeze(), cmap = use_colormap)

        if value_mean > hand_detection_limit:
            coords_scaled = coords_display * np.array([480 / 224, 640 / 224])
            tools.render_skeleton(prod_img, np.stack([coords_scaled[:, 1], coords_scaled[:, 0]], axis = 1), True,
                                  np.round(values, 3))

            for coord, value in zip(coords_scaled, values):
                prod_img = cv2.circle(prod_img, (int(coord[1]), int(coord[0])), 3, (0, 0, 0))

        cv2.imshow(win_name, prod_img)
        cv2.waitKey(1)

    cv2.destroyAllWindows()
    del cam

    return np.stack(sample_frames)


def element_diff(samples):
    if len(samples.shape) == 3:
        return np.diff(samples, axis = 0, append = 0)
    elif len(samples.shape) == 2:
        return np.diff(samples, axis = 1, append = 0)
    else:
        raise NotImplementedError()

def wrist_relative(samples):
    if len(samples.shape) == 3:
        res = samples - samples[:, 0][:, np.newaxis, :]
        res[:, 0, :] = samples[:, 0]
        return res
    elif len(samples.shape) == 2:
        res = samples - samples[0]
        res[0] = samples[0]
        return res
    else:
        raise NotImplementedError()


def run_app(model, action_manager, gesture_data):
    cam = RealsenseCamera({
            'file': camera_settings_file })
    time, depth_raw, rgb = cam.get_frame()  # camera warm up
    coords, vals = cv_from_frame(depth_raw, model)

    gesture_classifier = KNNClassifier(k = 2,
                                       batch_size = gesture_sample_length,
                                       sample_shape = coords.size,
                                       metric = 'manhattan'
                                       )
    X = []
    Y = []
    for i in range(5):
        X.append(np.zeros(coords.size * gesture_sample_length))
        Y.append(0)

    for idx, gesture in enumerate(gesture_data):
        for sample in gesture.samples:
            # X.append(sample.reshape(-1))
            # Y.append(idx + 1)

            #X.append(element_diff(sample).reshape(-1))
            #Y.append(idx + 1)

            X.append(wrist_relative(sample).reshape(-1))
            Y.append(idx + 1)

    gesture_classifier.set_train_data(X, Y)

    do_run = True
    display_results = True
    last_action_time = datetime.now()
    last_gesture = None
    last_coords = np.zeros(coords.shape)

    win_name = 'Self learning gesture control for automotive application...'
    cv2.namedWindow(win_name)

    one_euro = refiners.OneEuroFilter(freq = filter_param_freq, mincutoff = filter_param_mincutoff, beta = filter_param_beta)
    global_start_time = datetime.now()
    while do_run:

        start_time = datetime.now()
        time, depth_raw, rgb = cam.get_frame()  # camera warm up

        coords, vals = cv_from_frame(depth_raw, model)
        value_mean = np.mean(vals)

        if value_mean < hand_detection_limit:
            coords = np.zeros(coords.shape)

        coords = one_euro(coords, (datetime.now() - global_start_time).total_seconds())

        delta_coords = coords - last_coords
        last_coords = coords

        wrist_normalized_coords = wrist_relative(coords)

        # gesture_classifier.push_sample(coords.reshape(-1))
        #gesture_classifier.push_sample(delta_coords.reshape(-1))
        gesture_classifier.push_sample(wrist_normalized_coords.reshape(-1))

        gesture_prediction = gesture_classifier.predict()[0]

        if gesture_prediction > 0:
            gesture = gesture_data[gesture_prediction - 1]
            gesture_classifier.reset_queue()

            # debounce
            if ((datetime.now() - last_action_time).total_seconds() > 1) or \
                    ((datetime.now() - last_action_time).total_seconds() > 0.5 and not (last_gesture == gesture)):
                last_gesture = gesture
                last_action_time = datetime.now()
                action_manager.exec_action(gesture.action)
            else:
                print("Recognized gesture {} but not executing because {}".format(gesture, (datetime.now() - last_action_time).total_seconds()))
        else:
            gesture = None

        end_time = datetime.now()

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # esc
            do_run = False
        elif key == 100:  # d
            display_results = not display_results

        if display_results:
            if depth_raw.shape != (480, 640):
                depth_raw = cv2.resize(depth_raw, (640, 480))

            depth_raw = np.flip(depth_raw, 1)  # feels more natural
            coords_display = np.copy(coords)
            coords_display[:,1] = 224.0 - coords_display[:, 1]

            result_img = tools.colorize_cv(depth_raw.squeeze(), cmap = use_colormap)

            # Skeleton
            if value_mean > hand_detection_limit:
                coords_scaled = coords_display * np.array([480 / 224, 640 / 224])
                tools.render_skeleton(result_img, np.stack([coords_scaled[:, 1], coords_scaled[:, 0]], axis = 1), True,
                                      np.round(vals, 2))

            # FPS counter
            fps = (1 / (end_time - start_time).microseconds) * 1e6
            fps_text = "{:.01f} fps".format(fps)
            cv2.putText(result_img, fps_text, (10, 30), cv2.FONT_HERSHEY_PLAIN, fontScale = 0.75, color = (0, 0, 0),
                        thickness = 2)
            cv2.putText(result_img, fps_text, (10, 30), cv2.FONT_HERSHEY_PLAIN, fontScale = 0.75,
                        color = (255, 255, 255), thickness = 1)

            # Current Gesture Class
            last_name = "None" if last_gesture is None else last_gesture.name
            current_name = "None" if gesture_prediction == 0 else gesture.name
            gesture_text = "Last: {} ; Current: {}".format(last_name, current_name)
            cv2.putText(result_img, gesture_text, (320, 30), cv2.FONT_HERSHEY_PLAIN, fontScale = 0.75,
                        color = (0, 0, 0), thickness = 2)
            cv2.putText(result_img, gesture_text, (320, 30), cv2.FONT_HERSHEY_PLAIN, fontScale = 0.75,
                        color = (255, 255, 255), thickness = 1)

            # Current Gesture Probabilities
            class_probabilities = list(gesture_classifier.predict_proba())[0][1:]
            class_probabilities = dict(zip([gesture_data[i-1].name for i in gesture_classifier.class_names], class_probabilities))

            cv2.putText(result_img, str(class_probabilities), (320, 60), cv2.FONT_HERSHEY_PLAIN, fontScale = 0.75,
                        color = (0, 0, 0), thickness = 2)
            cv2.putText(result_img, str(class_probabilities), (320, 60), cv2.FONT_HERSHEY_PLAIN, fontScale = 0.75,
                        color = (255, 255, 255), thickness = 1)

        else:
            result_img = np.zeros((480, 640))
            cv2.putText(result_img, "Display deactivated!\nPress d to continue displaying results!", (10, 30),
                        cv2.FONT_HERSHEY_PLAIN, fontScale = 0.75, color = (255, 255, 255), thickness = 1)

        cv2.imshow(win_name, result_img)

    del cam
    cv2.destroyAllWindows()


def main(argv):
    gesture_data = []
    if os.path.exists('gesture_data.json'):
        gesture_data = deserialize_to_gesture_collection('gesture_data.json')

    if not os.path.exists(pose_model_path):
        logger.error("Pose estimation model could not be found at {}".format(pose_model_path))
        return

    model = tf.keras.models.load_model(pose_model_path, compile = False)
    model.predict(np.zeros((1, 224, 224, 1)))  # warm up to prevent lag later

    action_manager = app_framework.actions.ActionManager()

    control_center = app_framework.gui.MainWindow(action_manager,
                                                  sample_record_callback = lambda: record_sample(model),
                                                  save_gestures_callback = lambda
                                                      gestures: serialize_gesture_collection(gestures,
                                                                                             'gesture_data.json'),
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
    main(None)
    del tf
    print("App end!")

    from sys import exit



    exit(0)
