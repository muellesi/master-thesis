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
        full_ds = full_ds.map(lambda img, skel:  (tf.image.resize(img, tf.constant([img_dim[0], img_dim[1]], dtype=tf.dtypes.int32)), skel))
    #train_ds = full_ds.filter(lambda path, _, _: any(tf.strings.))
    return full_ds
