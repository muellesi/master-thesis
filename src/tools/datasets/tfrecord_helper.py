import tensorflow as tf
import numpy as np



# Code from https://www.tensorflow.org/tutorials/load_data/tfrecord
# The following functions can be used to convert a value to a type compatible
# with tf.Example.

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    if isinstance(value, np.ndarray):
        value = value.tolist()
    elif not isinstance(value, list):
        value = [value]

    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    if isinstance(value, np.ndarray):
        value = value.tolist()
    elif not isinstance(value, list):
        value = [value]

    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def make_serialized_example(feature):
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def decode_serialized_example(serialized_example):
    example_proto = tf.train.Example.FromString(serialized_example)


def make_standard_pose_record(idx, image, image_width, image_height, skeleton):
    skeleton = skeleton.reshape((-1))
    assert (skeleton.shape[0] == 21 * 3)
    feature = {
            'index': _int64_feature(idx),
            'depth': _bytes_feature(image),
            'image_width': _int64_feature(image_width),
            'image_height': _int64_feature(image_height),
            'skeleton': _float_feature(skeleton.reshape((-1)))
            }
    return make_serialized_example(feature)


def parse_standard_pose_record(serialized_record):
    feature_description = {
            'index': tf.io.FixedLenFeature([], tf.int64),
            'depth': tf.io.FixedLenFeature([], tf.string),
            'image_width': tf.io.FixedLenFeature([], tf.int64),
            'image_height': tf.io.FixedLenFeature([], tf.int64),
            'skeleton': tf.io.FixedLenFeature([21 * 3], tf.float32)
            }
    parsed = tf.io.parse_single_example(serialized_record, feature_description)
    return parsed['index'], parsed['depth'], parsed['image_width'], parsed['image_height'], parsed['skeleton']
