""""
    Dataset source: https://jonathantompson.github.io/NYU_Hand_Pose_Dataset.htm#download
    "Real-Time Continuous Pose Recovery of Human Hands Using Convolutional Networks"
    Jonathan Tompson, Murphy Stein, Yann Lecun and Ken Perlin.
    TOG'14 (Presented at SIGGRAPH'14)
"""

import argparse
import math
import os
import shutil

import cv2
import numpy as np
import progressbar
import scipy.io as sio
import tensorflow as tf

import datasets.tfrecord_helper as tfrh
import tools
import datasets.util

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)

__logger = tools.get_logger(__name__, do_file_logging = False)


def load_and_reencode(binfile):
    # Quote from https://jonathantompson.github.io/NYU_Hand_Pose_Dataset.htm#download:
    # "Note: In each depth png file the top 8 bits of depth are packed into the green channel and the lower 8 bits into blue."
    img = cv2.imread(binfile, cv2.IMREAD_UNCHANGED)
    img = img[:, :, 2] + img[:, :, 1] * 256.0
    img = np.expand_dims(img, axis = 2)
    success, encoded = cv2.imencode(".png", img.astype(np.uint16))
    return encoded.tobytes()


def prepare_dataset(dataset_location, image_width, image_height):
    """"
        Serializes the nyu dataset at dataset_location to tfrecord files at dataset_location/tfrecord_data
    """
    sample_types = ['train', 'validation', 'test']
    sample_start = [0, 0,
                    2440]  # use all training samples for training, from test folder use 1/3 for validation, 2/3 for test
    sample_end = [72757, 2440, 8252]
    num_chunks = [20, 1, 2]
    useful_indices = [0, 1, 3, 4, 6, 7, 9, 10, 12, 13, 15, 16, 18, 19, 21, 22, 24, 25, 26, 28, 35]  # for palm : , 34]
    reordered_indices = [20, 19, 15, 11, 7, 3, 18, 17, 16, 14, 13, 12, 10, 9, 8, 6, 5, 4, 2, 1,
                         0]  # fhad joint ordering, see serialize_fhad_data.py

    tfrecord_basepath = os.path.join(dataset_location,
                                     "tfrecord_data-{}_{}".format(image_width,
                                                                  image_height))
    if os.path.exists(tfrecord_basepath):
        shutil.rmtree(tfrecord_basepath)

    os.makedirs(tfrecord_basepath)

    progbar = None
    for type_idx, subset in enumerate(sample_types):
        subfolder = 'train' if subset == 'train' else 'test'
        base_path = os.path.join(dataset_location, subfolder)
        chunk_size = math.ceil((sample_end[type_idx] - sample_start[type_idx]) / num_chunks[type_idx])
        __logger.info(
                "Creating {} dataset from indices {} to {} in subfolder {}".format(subset, sample_start[type_idx],
                                                                                   sample_end[type_idx], subfolder))

        matfile = sio.loadmat(os.path.join(base_path, "joint_data.mat"))

        joint_xyz = matfile["joint_xyz"]
        joint_xyz = np.array(joint_xyz[0])  # throw away data for Kinect #2 & #3, only keep #1
        joint_xyz = joint_xyz.take(useful_indices, 1)
        joint_xyz = joint_xyz[:, reordered_indices, :]

        if progbar: del progbar
        progbar = progressbar.ProgressBar(widgets = [progressbar.Bar(), progressbar.Percentage(), progressbar.ETA()])
        progbar.start(max_value = sample_end[type_idx] - sample_start[type_idx])

        chunk = 0
        for i in range(sample_start[type_idx], sample_end[type_idx]):
            skel = joint_xyz[i]
            skel = skel.reshape((21, -1))
            skel = skel.dot(np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]]))  # y axis is somehow inverted in nyu dataset
            skel = skel.reshape((-1))
            img_path = '{}/depth_1_{:07d}.png'.format(base_path, i + 1)

            img = load_and_reencode(img_path)
            if 480 != image_height or 640 != image_width:
                img_decoded = tf.image.decode_png(img, channels = 1,
                                                  dtype = tf.dtypes.uint16)
                img_decoded = tf.image.resize(img_decoded,
                                              [image_height, image_width])
                img_decoded = tf.cast(img_decoded,
                                      dtype = tf.dtypes.uint16)
                img_encoded = tf.image.encode_png(img_decoded)
                img = img_encoded

            if i % chunk_size == 0 or i == sample_end[type_idx] - 1:
                # write data to chunk file
                filename = "{}_{}.tfrecord".format(subset, chunk)
                record_writer = tf.io.TFRecordWriter(os.path.join(tfrecord_basepath, filename))
                chunk += 1

            cam_intr = np.array([
                    [588.036865, 0.0, 320],
                    [0.0, 587.075073, 240],
                    [0.0, 0.0, 1.0]
                    ])

            skel_2d = tools.project_2d(skel.reshape(21, 3), cam_intr)

            if 480 != image_height or 640 != image_width:
                skel_2d = skel_2d.dot(np.array([[image_width / 640.0, 0.0],
                                                [0.0, image_height / 480.0]]))

            skel_maps = datasets.util.pts_to_confmaps(skel_2d,
                                                      image_width,
                                                      image_height,
                                                      sigma = 10)
            skel_maps = skel_maps * 2 ** 16
            skel_maps = skel_maps.astype(np.uint16)
            skel_maps_encoded = [
                    cv2.imencode(".png", m[:, :, np.newaxis])[1].tostring() for
                    m in skel_maps]


            tfrecord_str = tfrh.make_standard_pose_record(i,
                                                          img,
                                                          image_width,
                                                          image_height,
                                                          skel,
                                                          skel_maps_encoded)

            record_writer.write(tfrecord_str)

            if i % 10 == 0:
                progbar.update(i - sample_start[type_idx])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
            '--root',
            type = str,
            default = 'E:\\MasterDaten\\Datasets\\NYU\\nyu_hand_dataset_v2\\dataset',
            help = 'Root path for the dataset'
            )
    parser.add_argument(
            '--image_width',
            type = str,
            default = 224,
            help = 'Width of input images and skeleton maps'
            )
    parser.add_argument(
            '--image_height',
            type = str,
            default = 224,
            help = 'Height of input images and skeleton maps'
            )
    args = parser.parse_args()

    prepare_dataset(args.root, args.image_width, args.image_height)
