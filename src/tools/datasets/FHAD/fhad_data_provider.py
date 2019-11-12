import tensorflow as tf
import os
import numpy as np
import math
import h5py
import scipy.io as sio
import shutil
import cv2
import progressbar
import tools
import tools.datasets.tfrecord_helper as tfrh
import pathlib

__logger = tools.get_logger(__name__, do_file_logging=False)


def prepare_dataset(dataset_location):
    """"
        Serializes the FHAD dataset at dataset_location to tfrecord files at dataset_location/tfrecord_data
    """
    skeleton_location = os.path.join(dataset_location, "Hand_pose_annotation_v1")
    img_basepath = os.path.join(dataset_location, "Video_files")
    tfrecord_basepath = os.path.join(dataset_location, "tfrecord_data")

    train_subjects = ["Subject_1", "Subject_3", "Subject_4"]
    validate_subjects = ["Subject_2"]

    skeleton_files = [file for file in pathlib.Path(skeleton_location).rglob("skeleton.txt")]

    if os.path.exists(tfrecord_basepath):
        shutil.rmtree(tfrecord_basepath)

    os.makedirs(tfrecord_basepath)

    train_skel_files = []
    val_skel_files = []
    test_skel_files = []
    for skeleton_file in skeleton_files:
        relative_path = pathlib.Path(os.path.relpath(str(skeleton_file.absolute()), skeleton_location))
        subject = relative_path.parts[0]
        if subject in train_subjects:
            train_skel_files.append(skeleton_file)
        elif subject in validate_subjects:
            val_skel_files.append(skeleton_file)
        else:
            test_skel_files.append(skeleton_file)

    all_skel_files = np.array([train_skel_files, val_skel_files, test_skel_files])

    subsets = ["train", "validation", "test"]
    max_chunk_size = 5000

    progbar = None

    for idx, subset in enumerate(subsets):
        __logger.info("Creating serialized files for {} set".format(subset))
        sample_idx = 0
        chunk_idx = 0

        if progbar: del progbar
        progbar = progressbar.ProgressBar(widgets=[progressbar.Bar(), progressbar.Percentage(), progressbar.ETA()])
        progbar.start(max_value=len(all_skel_files[idx]))

        for file_idx, file in enumerate(all_skel_files[idx]):

            skels_world_coords = np.loadtxt(file)
            specific_img_basepath = os.path.join(img_basepath, file.relative_to(skeleton_location).parent, "depth")

            for line in skels_world_coords:
                frame_index = int(line[0])

                skeleton = line[1:]
                skeleton = skeleton.reshape((21, -1))
                skeleton = tools.skeleton_renderer.project_world_to_cam(skeleton, get_camera_extrinsics())
                skeleton = skeleton.reshape((-1))

                img_filename = "depth_{:04d}.png".format(frame_index)
                img_path = os.path.join(specific_img_basepath, img_filename)

                with tf.io.gfile.GFile(img_path, 'rb') as fid:
                    img = fid.read()

                if sample_idx % max_chunk_size == 0:
                    # write data to chunk file
                    filename = "{}_{}.tfrecord".format(subset, chunk_idx)
                    record_writer = tf.io.TFRecordWriter(os.path.join(tfrecord_basepath, filename))
                    chunk_idx += 1

                tfrecord_str = tfrh.make_standard_pose_record(sample_idx, img, 640, 480, skeleton)
                record_writer.write(tfrecord_str)
                sample_idx += 1

            progbar.update(file_idx)


@tf.function
def __decode_img(index, depth, img_width, img_height, skeleton):
    img = tf.image.decode_png(depth, channels=1, dtype=tf.uint16)
    return img, skeleton


@tf.function
def __open_tf_record(filename):
    tfrecord = tf.data.TFRecordDataset(filename)
    return tfrecord.map(tfrh.parse_standard_pose_record)


def get_dataset(dataset_location, subset='train'):
    """"
        Returns a tf.data.DataSet for the FHAD dataset

        Dataset source: https://guiggh.github.io/publications/first-person-hands/
        "First-Person Hand Action Benchmark with RGB-D Videos and 3D Hand Pose Annotations"
        Garcia-Hernando, Guillermo and Yuan, Shanxin and Baek, Seungryul and Kim, Tae-Kyun.
        Proceedings of Computer Vision and Pattern Recognition ({CVPR}), 2018

        :param subset: 'train', 'validation' or 'test'
        :param dataset_location: root path to the FHAD dataset
    """
    tfrecord_basepath = os.path.abspath(os.path.join(dataset_location, 'tfrecord_data'))

    if not os.path.exists(tfrecord_basepath):
        prepare_dataset(dataset_location)

    filenames = np.array([os.path.abspath(os.path.join(tfrecord_basepath, f)) for f in os.listdir(tfrecord_basepath) if f.startswith(subset)])

    dataset = (tf.data.Dataset.from_tensor_slices(filenames).cache()
               .flat_map(__open_tf_record)
               .map(__decode_img, num_parallel_calls=tf.data.experimental.AUTOTUNE))

    return dataset


def get_joint_names():
    """
    Returns a list of joint names, Ordered the way they are ordered in the FHAD dataset,
    see https://github.com/guiggh/hand_pose_action#hand-pose-data
    :return: A list of joint names
    """
    return ["wrist", "tmcp", "imcp", "mmcp", "rmcp", "pmcp", "tpip", "tdip", "ttip", "ipip", "idip",
            "itip", "mpip", "mdip", "mtip", "rpip", "rdip", "rtip", "ppip", "pdip", "ptip"]


def get_column_names():
    """
    Returns a list of joint coordinate names (format: wrist_x, wrist_y, wrist_z, ...),
    ordered the same as in the FHAD dataset, see https://github.com/guiggh/hand_pose_action#hand-pose-data
    :return: A list of joint coordinate names
    """
    joint_names = get_joint_names()
    return [["{}_{}".format(joint, axis) for axis in ["x", "y", "z"]] for joint in joint_names]


def get_camera_intrinsics(use_cam_intr='depth'):
    """
    Returns the respective (rgb/depth) camera's intrinsic matrix
    source: https://github.com/guiggh/hand_pose_action/blob/master/load_example.py#L147
    :param use_cam_intr: One of 'depth' or 'rgb'. Switches to the correct camera intrinsics, according to https://github.com/guiggh/hand_pose_action#camera-parameters
    :return: camera intrinsics
    """
    # see https://github.com/guiggh/hand_pose_action/blob/master/load_example.py#L113

    if use_cam_intr == 'rgb':
        cam_intr = np.array([[1395.749023, 0, 935.732544],
                             [0, 1395.749268, 540.681030],
                             [0, 0, 1]])
    else:
        cam_intr = np.array([[475.065948, 0, 315.944855],
                             [0, 475.065857, 245.287079],
                             [0, 0, 1]])

    return cam_intr


def get_camera_extrinsics():
    """
    Returns the camera's extrinsic matrix
    source: https://github.com/guiggh/hand_pose_action/blob/master/load_example.py#L147
    :return: camera extrinsics
    """
    cam_extr = np.array(
            [[0.999988496304, -0.00468848412856, 0.000982563360594, 25.7],
             [0.00469115935266, 0.999985218048, -0.00273845880292, 1.22],
             [-0.000969709653873, 0.00274303671904, 0.99999576807, 3.902],
             [0, 0, 0, 1.0]])
    return cam_extr
