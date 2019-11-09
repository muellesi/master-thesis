import tensorflow as tf
import os
import numpy as np



@tf.function
def __parse_skeleton(skeleton_data_line, file_path):
    data_elems = tf.strings.split(skeleton_data_line, " ")
    return file_path, data_elems[0], tf.strings.to_number(data_elems[1:], out_type=tf.dtypes.float32)


@tf.function
def __read_skeleton_file(file_path):
    skeleton_vals = tf.data.TextLineDataset(file_path).map(lambda x: __parse_skeleton(x, file_path))
    return skeleton_vals


@tf.function
def __get_corresponding_images(skeleton_path, frame_index, skeleton):
    video_path = tf.strings.regex_replace(skeleton_path, "Hand_pose_annotation_v1", "Video_files")
    path_length = tf.strings.length(video_path)
    video_path = tf.strings.substr(video_path, 0, path_length - len("skeleton.txt"))
    path_add = tf.constant("depth/depth_")
    file_ending = tf.constant(".png")
    img_path = tf.strings.join([video_path, path_add, frame_index, file_ending])
    return tf.io.decode_png(tf.io.read_file(img_path), dtype=tf.dtypes.uint16), skeleton


def get_dataset(path, img_dim=None):
    """"
        Returns a tf.data.DataSet for the FHAD dataset
        Dataset source: https://arxiv.org/abs/1704.02463 "First-Person Hand Action Benchmark with RGB-D Videos and 3D Hand Pose Annotations"
        :param path: root path to the dataset
        :param img_dim: tuple (height, width) of the requested data set
    """
    list_ds = tf.data.Dataset.list_files(os.path.join(path, '*/skeleton.txt'))
    skeleton_data = list_ds.flat_map(__read_skeleton_file)
    full_ds = skeleton_data.map(__get_corresponding_images)
    if img_dim is not None:
        full_ds = full_ds.map(lambda img, skel: (tf.image.resize(img, tf.constant([img_dim[0], img_dim[1]], dtype=tf.dtypes.int32)), skel))
    # train_ds = full_ds.filter(lambda path, _, _: any(tf.strings.))
    return full_ds


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


def project_skeleton(skel, use_cam_intr='depth'):
    """"
    Projects a given FHAD skeleton from world coordinates to camera coordinates/into the 2D plane
    source: https://github.com/guiggh/hand_pose_action/blob/master/load_example.py#L147
    :param skel: List of skeleton joint coordinates (see get_column_names())
    :param use_cam_intr: One of 'depth' or 'rgb'. Switches to the correct camera intrinsics, according to https://github.com/guiggh/hand_pose_action#camera-parameters
    :return: tuple (skel_camcoords, skel_proj_2d)
    """
    # see https://github.com/guiggh/hand_pose_action/blob/master/load_example.py#L113
    cam_extr = np.array(
            [[0.999988496304, -0.00468848412856, 0.000982563360594, 25.7],
             [0.00469115935266, 0.999985218048, -0.00273845880292, 1.22],
             [-0.000969709653873, 0.00274303671904, 0.99999576807, 3.902],
             [0, 0, 0, 1.0]])

    if use_cam_intr == 'rgb':
        cam_intr = np.array([[1395.749023, 0, 935.732544],
                             [0, 1395.749268, 540.681030],
                             [0, 0, 1]])
    else:
        cam_intr = np.array([[475.065948, 0, 315.944855],
                             [0, 475.065857, 245.287079],
                             [0, 0, 1]])

    skel_hom = np.concatenate([skel, np.ones([skel.shape[0], 1])], 1)
    skel_camcoords = cam_extr.dot(skel_hom.transpose()).transpose()[:, :3].astype(np.float32)

    skel_hom2d = np.array(cam_intr).dot(skel_camcoords.transpose()).transpose()
    skel_proj = (skel_hom2d / skel_hom2d[:, 2:])[:, :2]

    return skel_camcoords, skel_proj
