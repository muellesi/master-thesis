import os

import cv2
import numpy as np
import tensorflow as tf

import app_framework.actions
import app_framework.gui.main_window
import datasets.util
import tools
from app_framework.gesture_save_file import deserialize_to_gesture_collection
from app_framework.gesture_save_file import serialize_gesture_collection
from tools import RealsenseCamera



logger = tools.get_logger("MainApp")
configuration = None

last_frame_time = None
last_frame_depth = None
last_frame_rgb = None

pose_model_path = 'E:\\Google Drive\\UNI\\Master\\Thesis\\src\\data\\pose_est\\2d\\ueber_weihnachten\\pose_est_refined.hdf5'
gesture_sample_length = 90


def twod_argmax(val):
    maxy = tf.argmax(tf.reduce_max(val, axis = 2), 1)
    maxx = tf.argmax(tf.reduce_max(val, axis = 1), 1)
    maxs = tf.stack([maxy, maxx], axis = 2)
    maxs = tf.cast(maxs, dtype = tf.dtypes.float32)
    return maxs


def record_sample(model):
    display_size = (640, 480)
    countdown_font_scale = 5
    countdown_font_thickness = 2
    print("recording sample...")
    cam = RealsenseCamera({
            'file': 'E:\\Google Drive\\UNI\\Master\\Thesis\\src\\realsense_settings.json' })

    # Open Window
    win_name = "Sample record..."
    cv2.namedWindow(win_name)

    # Show Countdown

    from datetime import datetime
    from datetime import timedelta

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

    sample_frames = []

    for i in range(gesture_sample_length):
        # Get Frame
        time, depth_raw, rgb = cam.get_frame()  # camera warm up
        depth = cv2.resize(depth_raw, (224, 224))
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

        value_norm = np.linalg.norm(values)
        value_max = np.max(values)
        value_min = np.min(values)

        sample_frames.append(coords)

        prod_img = tools.colorize_cv(depth_raw.squeeze())
        if value_norm > 0.5:
            import colorsys
            coords_scaled = coords * np.array([480 / 224, 640 / 224])
            tools.render_skeleton(prod_img, np.stack([coords_scaled[:, 1], coords_scaled[:, 0]], axis=1), True, values)

            for coord, value in zip(coords * np.array([480 / 224, 640 / 224]), values):
                c = colorsys.hls_to_rgb(0.375 * value, 0.5, 0.5)
                color = (c[2], c[1], c[0])
                prod_img = cv2.circle(prod_img, (int(coord[1]), int(coord[0])), 3, color)

        cv2.imshow(win_name, prod_img)
        cv2.waitKey(1)

    cv2.destroyAllWindows()
    return np.stack(sample_frames)


def run_app(gesture_data):
    raise NotImplementedError()


def main(argv):
    gesture_data = []
    if os.path.exists('gesture_data.json'):
        gesture_data = deserialize_to_gesture_collection('gesture_data.json')

    if not os.path.exists(pose_model_path):
        logger.error("Pose estimation model could not be found at {}".format(pose_model_path))
        return

    model = tf.keras.models.load_model(pose_model_path, compile = False)

    action_manager = app_framework.actions.ActionManager()

    for i in range(20):
        g = app_framework.GestureItem(name = 'gesture {}'.format(i),
                                      samples = [],
                                      action =
                                      app_framework.actions.HelloWorldAction().get_name())
        gesture_data.append(g)

    control_center = app_framework.gui.MainWindow(action_manager,
                                                  sample_record_callback = lambda: record_sample(model),
                                                  save_gestures_callback = lambda
                                                      gestures: serialize_gesture_collection(gestures,
                                                                                             'gesture_data.json'),
                                                  main_app_callback = run_app)
    control_center.set_gestures(gesture_data)
    while control_center.alive:
        control_center.update()

    return


if __name__ == "__main__":
    main(None)
