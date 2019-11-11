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



__logger = tools.get_logger(__name__, do_file_logging=False)


def prepare_dataset(dataset_location):
    """"
        Serializes the nyu dataset at dataset_location to tfrecord files at dataset_location/tfrecord_data
    """
    sample_types = ['train', 'validation', 'test']
    sample_start = [0, 0, 2440]  # use all training samples for training, from test folder use 1/3 for validation, 2/3 for test
    sample_end = [72757, 2440, 8252]
    num_chunks = [20, 1, 2]
    useful_indices = [0, 1, 3, 4, 6, 7, 9, 10, 12, 13, 15, 16, 18, 19, 21, 22, 24, 25, 26, 28, 35]  # for palm : , 34]
    reordered_indices = [20, 19, 15, 11, 7, 3, 18, 17, 16, 14, 13, 12, 10, 9, 8, 6, 5, 4, 2, 1, 0]  # fhad joint ordering, see fhad_data_provider.py

    tfrecord_basepath = os.path.abspath(os.path.join(dataset_location, 'tfrecord_data'))

    if os.path.exists(tfrecord_basepath):
        shutil.rmtree(tfrecord_basepath)

    os.makedirs(tfrecord_basepath)

    progbar = None
    for type_idx, subset in enumerate(sample_types):
        subfolder = 'train' if subset == 'train' else 'test'
        base_path = os.path.join(dataset_location, subfolder)
        chunk_size = math.ceil((sample_end[type_idx] - sample_start[type_idx]) / num_chunks[type_idx])
        __logger.info(
                "Creating {} dataset from indices {} to {} in subfolder {}".format(subset, sample_start[type_idx], sample_end[type_idx], subfolder))

        matfile = sio.loadmat(os.path.join(base_path, "joint_data.mat"))

        joint_xyz = matfile["joint_xyz"]
        joint_xyz = np.array(joint_xyz[0])  # throw away data for Kinect #2 & #3, only keep #1
        joint_xyz = joint_xyz.take(useful_indices, 1)
        joint_xyz = joint_xyz[:, reordered_indices, :]

        if progbar: del progbar
        progbar = progressbar.ProgressBar(widgets=[progressbar.Bar(), progressbar.Percentage(), progressbar.ETA()])
        progbar.start(max_value=sample_end[type_idx] - sample_start[type_idx])

        chunk = 0
        for i in range(sample_start[type_idx], sample_end[type_idx]):
            skel = joint_xyz[i]
            skel = skel.reshape((21, -1))
            skel = skel.dot(np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]]))  # y axis is somehow inverted in nyu dataset
            skel = skel.reshape((-1))
            img_path = '{}/depth_1_{:07d}.png'.format(base_path, i + 1)

            with tf.io.gfile.GFile(img_path, 'rb') as fid:
                img = fid.read()

            if i % chunk_size == 0 or i == sample_end[type_idx] - 1:
                # write data to chunk file
                filename = "{}_{}.tfrecord".format(subset, chunk)
                record_writer = tf.io.TFRecordWriter(os.path.join(tfrecord_basepath, filename))
                chunk += 1

            tfrecord_str = tfrh.make_standard_pose_record(i, img, 640, 480, skel)
            record_writer.write(tfrecord_str)

            if i % 10 == 0:
                progbar.update(i - sample_start[type_idx])


@tf.function
def __decode_img(index, depth, img_width, img_height, skeleton):
    img = tf.image.decode_png(depth, channels=3)
    # Quote from https://jonathantompson.github.io/NYU_Hand_Pose_Dataset.htm#download:
    # "Note: In each depth png file the top 8 bits of depth are packed into the green channel and the lower 8 bits into blue."
    img = tf.dtypes.cast(img, tf.int16)
    img = (img[:, :, 2] + img[:, :, 1] * tf.constant(256, dtype=tf.int16))
    img = tf.expand_dims(img, 2)
    return img, skeleton


@tf.function
def __open_tf_record(filename):
    tfrecord = tf.data.TFRecordDataset(filename)
    return tfrecord.map(tfrh.parse_standard_pose_record)


def get_dataset(dataset_location, subset="train", img_dim=None):
    """"
        Returns a tf.data.DataSet for the NYU dataset

        Dataset source: https://jonathantompson.github.io/NYU_Hand_Pose_Dataset.htm#download
        "Real-Time Continuous Pose Recovery of Human Hands Using Convolutional Networks"
        Jonathan Tompson, Murphy Stein, Yann Lecun and Ken Perlin.
        TOG'14 (Presented at SIGGRAPH'14)

        :param subset: 'train', 'validation' or 'test'
        :param dataset_location: root path to the nyu dataset
        :param img_dim: tuple (height, width) of the requested data set
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
    Returns a list of joint names, Ordered the way they are ordered in the FHAD dataset (prepare_dataset takes care of that),
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


def get_camera_intrinsics():
    """"
    from xyz_to_uvd script in dataset: https://jonathantompson.github.io/NYU_Hand_Pose_Dataset.htm#download
    :return: Camera intrinsics of the NYU dataset
    """
    half_res_x = 640 / 2
    half_res_y = 480 / 2
    coeff_x = 588.036865
    coeff_y = 587.075073

    cam_intr = np.array([[coeff_x, 0, half_res_x],
                         [0, coeff_y, half_res_y],
                         [0, 0, 1]])
    return cam_intr


def get_camera_extrinsics():
    """
    Returns the camera's extrinsic matrix
    source: https://github.com/guiggh/hand_pose_action/blob/master/load_example.py#L147
    :return: camera extrinsics
    """
    raise NotImplementedError("No extrinsics known for NYU dataset!")
