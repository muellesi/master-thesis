import numpy as np
import tensorflow as tf
from scipy.stats import multivariate_normal

import tools



@tf.function
def scale_image(img, new_shape):
    img = tf.image.resize(img, tf.constant(new_shape, dtype = tf.dtypes.int32))
    img = tf.cast(img, dtype = tf.float32)
    return img


@tf.function
def scale_clip_image_data(img, scale):
    img = img * tf.constant(scale, dtype = tf.float32)
    img = tf.clip_by_value(img, clip_value_min = 0.0,
                           clip_value_max = 1.0)  # ignore stuff more than
    # 2.5m away.
    return img


@tf.function
def add_random_noise(img):
    noise = tf.random.normal(shape = tf.shape(img), mean = 0.0, stddev = 0.01,
                             dtype = tf.dtypes.float32)
    return img + noise


def batch_shuffle_prefetch(ds, batch_size):
    ds = ds.shuffle(batch_size * 20)
    ds = ds.batch(batch_size = batch_size)
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
    return ds


def eukl_distance(distance_x, distance_y):
    return np.sqrt(np.square(distance_x) + np.square(distance_y))


def gaussian(x, mu, sig):
    value = np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
    return value


def pts_to_confmaps(pts, map_width, map_height, sigma = 5):
    # conversion of https://github.com/talmo/leap/blob/master/leap
    # /pts2confmaps.m
    """
    pts_to_confmaps Generate confidence maps centered at specified points.

    :param pts: N x 2 or cell array of {N1 x 2, N2 x 2, ...}, where each
    cell will correspond to a single channel to create multipoint confidence
    maps
    :param height: image_height
    :param width: image_width
    :param sigma: filter size (default: 5)
    :return: conf_maps
    """
    conf_maps = np.zeros(
            shape = (int(pts.shape[0]), int(map_height), int(map_width)),
            dtype = np.float32)
    i = 0
    for pt in pts:
        x, y = pt
        pos = np.dstack(np.mgrid[0:map_height:1, 0:map_width:1])
        rv = multivariate_normal(mean = [y, x], cov = sigma)
        conf_maps[i] = rv.pdf(pos)
        i = i + 1

    return conf_maps


def skel_to_confmaps(skel, intr_matrix, map_width, map_height, skel_stretch_x,
                     skel_stretch_y, sigma = 5):
    skel = skel.reshape(-1, 3)
    skel = tools.project_2d(skel, intr_matrix)
    skel = skel.dot(np.array([[skel_stretch_x, 0], [0, skel_stretch_y]]))
    maps = pts_to_confmaps(skel, map_width, map_height, sigma)
    maps = np.transpose(maps, (1, 2, 0))
    return maps


def _load_img_from_path(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, channels = 1, dtype = tf.dtypes.uint16)
    return img


def make_img_ds_from_glob(glob_pattern, width, height, value_scale = None,
                          shuffle = True):
    ds = tf.data.Dataset.list_files(glob_pattern, shuffle = shuffle).cache()

    ds = ds.map(_load_img_from_path,
                num_parallel_calls = tf.data.experimental.AUTOTUNE)

    ds = ds.map(lambda img: scale_image(img, [height, width]),
                num_parallel_calls = tf.data.experimental.AUTOTUNE)

    if value_scale:
        ds = ds.map(lambda img: scale_clip_image_data(img, value_scale),
                    num_parallel_calls = tf.data.experimental.AUTOTUNE)

    return ds
