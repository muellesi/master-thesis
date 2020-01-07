import os

import numpy as np

import app_framework.actions
import app_framework.gui.main_window
import tools
from app_framework.gesture_save_file import deserialize_to_gesture_collection
from app_framework.gesture_save_file import serialize_gesture_collection



configuration = None

last_frame_time = None
last_frame_depth = None
last_frame_rgb = None


def overlay_skeleton(img, skel_cam_coord, skew_factor = None,
                     intrinsics = None, joint_names = None):
    skeleton_2d = tools.skeleton_renderer.project_2d(skel_cam_coord,
                                                     intrinsics)
    if skew_factor:
        skeleton_2d = skeleton_2d.dot(
                np.array([[skew_factor[1], 0], [0, skew_factor[0]]]))
    image2 = tools.image_colorizer.colorize_cv(img, 0.0, 1.0, 'viridis')
    tools.render_skeleton(image2, skeleton_2d, joint_names = joint_names)
    return image2


def record_sample():
    print("recording sample...")
    return np.zeros((90, 21, 2))




def run_app(gesture_data):
    raise NotImplementedError()


def main(argv):
    gesture_data = []
    if os.path.exists('gesture_data.json'):
        gesture_data = deserialize_to_gesture_collection('gesture_data.json')

    print(gesture_data)

    action_manager = app_framework.actions.ActionManager()

    for i in range(20):
        g = app_framework.GestureItem(name = 'gesture {}'.format(i),
                                      samples = [],
                                      action =
                                      app_framework.actions.HelloWorldAction().get_name())
        gesture_data.append(g)

    control_center = app_framework.gui.MainWindow(action_manager,
                                                  sample_record_callback = record_sample,
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
