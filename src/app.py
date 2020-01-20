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

pose_model_path = 'E:\\Google Drive\\UNI\\Master\\Thesis\\src\\data\\pose_est\\2d\\ueber_weihnachten\\pose_est_refined.hdf5'
camera_settings_file = 'E:\\Google Drive\\UNI\\Master\\Thesis\\src\\realsense_settings.json'
gesture_sample_length = 90
norm_limit = 0.5


def twod_argmax(val):
    maxy = tf.argmax(tf.reduce_max(val, axis = 2), 1)
    maxx = tf.argmax(tf.reduce_max(val, axis = 1), 1)
    maxs = tf.stack([maxy, maxx], axis = 2)
    maxs = tf.cast(maxs, dtype = tf.dtypes.float32)
    return maxs


def cv_from_frame(frame, model):
    depth = cv2.resize(frame, (224, 224))
    depth = datasets.util.scale_clip_image_data(depth, 1.0 / 1000.0)

    # Pose estimation
    depth = np.expand_dims(np.expand_dims(depth, 2), 0)
    res = model.predict(depth)

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

    for i in range(gesture_sample_length):
        # Get Frame
        time, depth_raw, rgb = cam.get_frame()  # camera warm up
        coords, values = cv_from_frame(depth_raw, model)

        value_norm = np.linalg.norm(values)

        if value_norm < norm_limit:
            coords = np.zeros(coords.shape)

        sample_frames.appendleft(coords)

        prod_img = tools.colorize_cv(depth_raw.squeeze())
        if value_norm > 0.5:
            coords_scaled = coords * np.array([480 / 224, 640 / 224])
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
        return np.diff(samples, axis = 0, append = [samples[-1, :]])
    elif len(samples.shape) == 2:
        return np.diff(samples, axis = 1, append = [samples[-1, :]])
    else:
        raise NotImplementedError()

def wrist_relative(samples):
    if len(samples.shape) == 3:
        return samples - samples[:, 0][:, np.newaxis, :]
    elif len(samples.shape) == 2:
        return samples - samples[0]
    else:
        raise NotImplementedError()


def run_app(model, action_manager, gesture_data):
    cam = RealsenseCamera({
            'file': camera_settings_file })
    time, depth_raw, rgb = cam.get_frame()  # camera warm up
    coords, vals = cv_from_frame(depth_raw, model)

    gesture_classifier = KNNClassifier(k = 3,
                                       batch_size = gesture_sample_length,
                                       sample_shape = coords.size
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

            X.append(element_diff(sample).reshape(-1))
            Y.append(idx + 1)

            # X.append(wrist_relative(sample).reshape(-1))
            # Y.append(idx + 1)

    gesture_classifier.set_train_data(X, Y)

    do_run = True
    display_results = True
    last_action_time = datetime.now()
    last_gesture = None
    last_coords = np.zeros(coords.shape)

    win_name = 'Self learning gesture control for automotive application...'
    cv2.namedWindow(win_name)

    while do_run:

        start_time = datetime.now()
        time, depth_raw, rgb = cam.get_frame()  # camera warm up

        coords, vals = cv_from_frame(depth_raw, model)
        value_norm = np.linalg.norm(vals)

        if value_norm < norm_limit:
            coords = np.zeros(coords.shape)

        delta_coords = coords - last_coords
        last_coords = coords

        wrist_normalized_coords = wrist_relative(coords)

        # gesture_classifier.push_sample(coords.reshape(-1))
        gesture_classifier.push_sample(delta_coords.reshape(-1))
        # gesture_classifier.push_sample(wrist_normalized_coords.reshape(-1))

        gesture_prediction = gesture_classifier.predict()[0]

        if gesture_prediction > 0:
            gesture = gesture_data[gesture_prediction - 1]
            gesture_classifier.reset_queue()

            # debounce
            if ((datetime.now() - last_action_time).microseconds > 600e3) or \
                    ((datetime.now() - last_action_time).microseconds > 10e3 and not (last_gesture == gesture)):
                last_gesture = gesture
                last_action_time = datetime.now()
                action_manager.exec_action(gesture.action)
        else:
            gesture = None

        end_time = datetime.now()

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            do_run = False
        elif key == 100:
            display_results = not display_results

        if display_results:
            result_img = tools.colorize_cv(depth_raw.squeeze())

            # Skeleton
            if value_norm > norm_limit:
                coords_scaled = coords * np.array([480 / 224, 640 / 224])
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
            class_probabilities = dict(gesture_classifier.predict_proba())
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
