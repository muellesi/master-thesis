import math
import numpy as np
import pandas as pd
from keras.utils import Sequence
import cv2
import os
import shutil
from .loggingutil import get_logger
import logging
# supposed directory structure:
# root
# - Video_files
#   - Subject_x
#       - gestureclass
#           - y
#               - color
#                   - color_0000.jpeg
#               - depth
#                   - depth_0000.png
# - Hand_pose_annotation_v1
#   - Subject_x
#       - gestureclass
#           - y
#               - skeleton.txt


class FHADDataProvider(Sequence):

    def __init__(self, data_dir, batch_size, img_dim_x=200, img_dim_y=200):

        self.logger = get_logger(name=__name__, do_file_logging=False)
        self.data_dir = data_dir

        self.video_dir = os.path.join(data_dir, "Video_files")
        self.label_dir = os.path.join(data_dir, "Hand_pose_annotation_v1")

        self.img_width = img_dim_x
        self.img_height = img_dim_y

        self.batch_size = batch_size

        self.image_files = []
        self.labels = []

        self.logger.info("Searching for data in {}...".format(self.data_dir))
        self.load_data_info()

        self.indices = np.arange(len(self.image_files))

    def load_data_info(self, use_checkpoint=True):
        save_dir = "data/pose/"
        image_save_path = os.path.join(save_dir, "images.txt")
        skeleton_save_path = os.path.join(save_dir, "skeletons.npy")

        if use_checkpoint and os.path.exists(image_save_path) and os.path.exists(skeleton_save_path):
            logging.warning("Loading from save, won't crawl data dir!")
            with open(image_save_path, "r") as f:
                self.image_files = [line for line in (x.strip() for x in f)]
            self.labels = np.load(skeleton_save_path)
        else:
            self.logger.info(
                "Searching for depth images in {}".format(self.video_dir))
            for subdir, dirs, files in os.walk(self.video_dir):
                if os.path.basename(subdir) == "depth":
                    try:
                        image_paths = [os.path.join(
                            subdir, filename) for filename in files if os.path.splitext(filename)[1] == ".png"]
                        relative_path = os.path.relpath(subdir, start=self.video_dir)[
                            :-len("depth")]
                        corresponding_label_dir = os.path.join(
                            self.label_dir, relative_path)
                        skeleton_data = pd.read_csv(os.path.join(
                            corresponding_label_dir, "skeleton.txt"), sep=" ", header=None)
                        skeleton_data = skeleton_data.drop(
                            columns=skeleton_data.columns[0])  # only keep real skeleton data
                        skeleton_data = skeleton_data.to_numpy()
                        if len(image_paths) != len(skeleton_data):
                            raise Exception("Number of images ({}) and number of labels ({}) are not equal for dataset in {}".format(
                                len(skeleton_data), len(image_paths), relative_path))
                    except Exception as e:
                        self.logger.exception(e)
                    else:
                        self.labels.extend(skeleton_data)
                        self.image_files.extend(image_paths)
                        self.logger.debug("Added {} images from {}".format(
                            len(image_paths), subdir))
                        self.logger.debug("Added {} labels from {}".format(
                            len(skeleton_data), os.path.join(corresponding_label_dir, "skeleton.txt")))

            self.logger.info(
                "Found {} depth images".format(len(self.image_files)))
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            with open(image_save_path, "w") as f:
                f.writelines(s + '\n' for s in self.image_files)
            np.save(skeleton_save_path, self.labels)
            self.logger.info("Saved found data to {}".format(save_dir))

    def __len__(self):
        return math.ceil(len(self.image_files) / self.batch_size)

    def __getitem__(self, idx):
        self.logger.info("Batch {} was requested".format(idx))
        batch_x = self.image_files[idx *
                                   self.batch_size: (idx + 1) * self.batch_size]
        batch_y = self.labels[idx *
                              self.batch_size: (idx + 1) * self.batch_size]
        imgs = np.expand_dims(np.array([cv2.resize(cv2.imread(filename, cv2.IMREAD_ANYCOLOR), (self.img_width, self.img_height))
                         for filename in batch_x]), axis=-1)
        return imgs, np.array(batch_y)
