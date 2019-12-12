import argparse
import sys
import cv2
import os
import matplotlib.pyplot as plt
import matplotlib
import dataclasses
import numpy as np

import json


import models.models as models
from models.knn_gesture_classifier import KNNClassifier


import tools
from datasets import SerializedDataset
from app_framework.gui.main_window import MainWindow
from app_framework.actions.action_manager import ActionManager


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


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def main(argv):
    with open('gesture_data.json', 'r') as f:
        gesture_data = json.load(f)

    print(gesture_data)

    action_manager = ActionManager()
    
    for i in range(20):
        g = GestureItem(name='gesture {}'.format(i), samples = [], action = None)
        gesture_data.append(GestureItem)

    control_center = MainWindow()
    while control_center.alive:
        control_center.update()

    return 



if __name__ == "__main__":
    main(None)
