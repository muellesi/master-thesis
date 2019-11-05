import math
import numpy as np
import pandas as pd
from keras.utils import Sequence
import cv2
import os
import shutil
import logging
import tensorflow as tf
import threading
from datetime import datetime
import sys

from numpy.core.multiarray import ndarray

from .loggingutil import get_logger

logger = get_logger(name=__name__, do_file_logging=False)


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

# preprocessing to shard files inspired by https://github.com/tensorflow/models/blob/master/research/inception/inception/data/build_image_data.py
def _int64_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_feature(value):
    if isinstance(value, np.ndarray):
        value = value.tolist()
    elif not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def wrap_proto(filename, image_buffer, skeleton, height, width):
    colorspace = 'GRAY'
    channels = 1
    image_format = 'PNG'

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': _int64_feature(height),
        'image/width': _int64_feature(width),
        'image/colorspace': _bytes_feature(tf.compat.as_bytes(colorspace)),
        'image/channels': _int64_feature(channels),
        'image/skeleton': _float_feature(skeleton.reshape([-1])),
        'image/format': _bytes_feature(tf.compat.as_bytes(image_format)),
        'image/filename': _bytes_feature(tf.compat.as_bytes(os.path.basename(filename))),
        'image/encoded': _bytes_feature(tf.compat.as_bytes(image_buffer))}))
    return example


class ImageDecoder(object):

    def __init__(self):
        self._sess = tf.Session()
        self._decode_depth_data = tf.placeholder(dtype=tf.string)
        self._decode_depth = tf.image.decode_png(self._decode_depth_data, dtype=tf.uint16)

    def decode_depth(self, image_data):
        image = self._sess.run(self._decode_depth,
                               feed_dict={self._decode_depth_data: image_data})
        assert len(image.shape) == 3
        assert image.shape[2] == 1
        return image


def process_single_image(filename: str, img_decoder: ImageDecoder):
    with tf.gfile.FastGFile(filename, 'rb') as f:
        image_data = f.read()
    image = img_decoder.decode_depth(image_data)
    height = image.shape[0]
    width = image.shape[1]
    return image_data, height, width


def process_images_batch(purpose, decoder, thread_index, ranges, filenames, skeletons, num_shards, out_dir):
    num_threads = len(ranges)
    assert not num_shards % num_threads
    num_shards_per_batch = int(num_shards / num_threads)

    shard_ranges = np.linspace(ranges[thread_index][0],
                               ranges[thread_index][1],
                               num_shards_per_batch + 1).astype(int)
    num_files_in_thread = ranges[thread_index][1] - ranges[thread_index][0]

    counter = 0
    for s in range(num_shards_per_batch):
        shard = thread_index * num_shards_per_batch + s
        output_filename = "{}-{:05d}-of-{:05d}".format(purpose, shard, num_shards)
        output_file = os.path.join(out_dir, output_filename)
        writer = tf.python_io.TFRecordWriter(output_file)

        shard_counter = 0
        files_in_shard = np.arange(shard_ranges[s], shard_ranges[s + 1], dtype=int)
        for i in files_in_shard:
            filename = filenames[i]
            skeleton = skeletons[i]

            try:
                image_buffer, height, width = process_single_image(filename, decoder)
            except Exception as e:
                print(e)
                logger.exception(e)
                print('SKIPPED: Unexpected error while decoding %s.' % filename)
                continue

            example = wrap_proto(filename, image_buffer, skeleton, height, width)
            writer.write(example.SerializeToString())
            shard_counter += 1
            counter += 1

            if not counter % 1000:
                print('%s [thread %d]: Processed %d of %d images in thread batch.' %
                      (datetime.now(), thread_index, counter, num_files_in_thread))
                sys.stdout.flush()

        writer.close()
        print('%s [thread %d]: Wrote %d images to %s' %
              (datetime.now(), thread_index, shard_counter, output_file))
        sys.stdout.flush()
        shard_counter = 0
    print('%s [thread %d]: Wrote %d images to %d shards.' %
          (datetime.now(), thread_index, counter, num_files_in_thread))
    sys.stdout.flush()


def process_image_list(purpose: str, filenames: list, skeletons: list, num_shard_files: int, out_dir: str,
                       num_threads: int = 8, ):
    assert len(filenames) == len(skeletons)

    spacing = np.linspace(0, len(filenames), num_threads + 1).astype(np.int)
    batch_ranges = []
    for i in range(len(spacing) - 1):
        batch_ranges.append([spacing[i], spacing[i + 1]])

    coord = tf.train.Coordinator()
    decoder = ImageDecoder()

    start_time = datetime.now()

    threads = []
    for thread_index in range(len(batch_ranges)):
        args = (purpose, decoder, thread_index, batch_ranges, filenames, skeletons, num_shard_files, out_dir)
        t = threading.Thread(target=process_images_batch, args=args)
        t.start()
        threads.append(t)

    coord.join(threads)  # wait for worker threads
    print("Took {} to write all {} images in data set to shard files.".format(start_time - datetime.now(),
                                                                              len(filenames)))
    sys.stdout.flush()


class FHADDataProvider(Sequence):

    def __init__(self, data_dir):

        # see https://github.com/guiggh/hand_pose_action/blob/master/load_example.py#L113
        self.cam_extr = np.array(
            [[0.999988496304, -0.00468848412856, 0.000982563360594, 25.7],
             [0.00469115935266, 0.999985218048, -0.00273845880292, 1.22],
             [-0.000969709653873, 0.00274303671904, 0.99999576807, 3.902],
             [0, 0, 0, 1.0]])

        self.cam_intr = np.array([[1395.749023, 0, 935.732544],
                                  [0, 1395.749268, 540.681030],
                                  [0, 0, 1]])

        self.train_subjects = [1, 3, 4]  # https://github.com/guiggh/hand_pose_action#hand-pose-estimation

        self.data_dir = data_dir

        self.video_dir = os.path.join(data_dir, "Video_files")
        self.label_dir = os.path.join(data_dir, "Hand_pose_annotation_v1")

        # see https://github.com/guiggh/hand_pose_action#hand-pose-data
        self.joint_names = ["wrist", "tmcp", "imcp", "mmcp", "rmcp", "pmcp", "tpip", "tdip", "ttip", "ipip", "idip",
                            "itip", "mpip", "mdip", "mtip", "rpip", "rdip", "rtip", "ppip", "pdip", "ptip"]

        self.location_names = [["{}_{}".format(finger, axis) for axis in ["x", "y", "z"]]
                               for finger in self.joint_names]

        logger.info("Skeleton joint data structure: {}".format(self.location_names))

        if not os.path.exists(os.path.join(self.data_dir, "preprocessed")):
            logger.info("Searching for data in {}...".format(self.data_dir))
            image_files, skeletons = self.load_data_info()
            self.indices = list(np.arange(len(image_files)))

            train_subjects = ["Subject_{x}".format(x=subj_num) for subj_num in self.train_subjects]

            train_indices = [index for index, file in enumerate(image_files) if any(subject in file for subject in train_subjects)]
            eval_indices = [index for index, file in enumerate(image_files) if not any(subject in file for subject in train_subjects)]
            train_set = [[image_files[i] for i in train_indices], [skeletons[i] for i in train_indices]]
            eval_set = [[image_files[i] for i in eval_indices], [skeletons[i] for i in eval_indices]]

            if not os.path.exists(os.path.join(data_dir, "preprocessed")):
                os.makedirs(os.path.join(data_dir, "preprocessed"))

            process_image_list("train", train_set[0], train_set[1], 8 * 10, os.path.join(data_dir, "preprocessed"), 8)
            process_image_list("eval", eval_set[0], eval_set[1], 8 * 10, os.path.join(data_dir, "preprocessed"), 8)

    def get_column_names(self):
        return self.column_names

    # source: https://github.com/guiggh/hand_pose_action/blob/master/load_example.py#L147
    def project_skeleton(self, skel):
        skel_hom = np.concatenate([skel, np.ones([skel.shape[0], 1])], 1)
        skel_camcoords = self.cam_extr.dot(
            skel_hom.transpose()).transpose()[:, :3].astype(np.float32)

        skel_hom2d = np.array(self.cam_intr).dot(skel_camcoords.transpose()).transpose()
        skel_proj = (skel_hom2d / skel_hom2d[:, 2:])[:, :2]

    def load_data_info(self, use_checkpoint=True):
        skeletons = []
        image_files = []
        save_dir = "data/pose/"
        image_save_path = os.path.join(save_dir, "images.txt")
        skeleton_save_path = os.path.join(save_dir, "skeletons.npy")

        logger.info(
            "Searching for depth images in {}".format(self.video_dir))
        for subdir, dirs, files in os.walk(self.video_dir):
            if os.path.basename(subdir) == "depth":
                try:
                    image_paths = [os.path.join(
                        subdir, filename) for filename in files if os.path.splitext(filename)[1] == ".png"]
                    relative_path = os.path.relpath(subdir, start=self.video_dir)[:-len("depth")]
                    corresponding_label_dir = os.path.join(self.label_dir, relative_path)

                    skeleton_vals = np.loadtxt(os.path.join(corresponding_label_dir, "skeleton.txt"))
                    skeleton_data = skeleton_vals[:, 1:].reshape(skeleton_vals.shape[0], 21,
                                                                 -1)  # only keep real skeleton data

                    if len(image_paths) != len(skeleton_data):
                        raise Exception(
                            "Number of images ({}) and number of labels ({}) are not equal for dataset in {}".format(
                                len(skeleton_data), len(image_paths), relative_path))
                except Exception as e:
                    logger.exception(e)
                else:
                    skeletons.extend(skeleton_data)
                    image_files.extend(image_paths)
                    logger.debug("Added {} images from {}".format(
                        len(image_paths), subdir))
                    logger.debug("Added {} labels from {}".format(
                        len(skeleton_data), os.path.join(corresponding_label_dir, "skeleton.txt")))

        logger.info("Found {} depth images".format(len(image_files)))
        assert len(image_files) == len(skeletons)
        return image_files, skeletons

    # def __len__(self):
    #     return math.ceil(len(self.image_files) / self.batch_size)
    #
    # def __getitem__(self, idx):
    #     logger.info("Batch {} was requested".format(idx))
    #     batch_x = self.image_files[idx *
    #                                self.batch_size: (idx + 1) * self.batch_size]
    #     batch_y = self.skeletons[idx *
    #                              self.batch_size: (idx + 1) * self.batch_size]
    #
    #     imgs = np.expand_dims(
    #         np.array([cv2.resize(cv2.imread(filename, cv2.IMREAD_ANYCOLOR), (self.img_width, self.img_height))
    #                   for filename in batch_x]), axis=-1)
    #     return imgs, np.array(batch_y)
