import json

import numpy as np
import tensorflow as tf

import datasets.util
from datasets import SerializedDataset
from datasets.tfrecord_helper import decode_confmaps



net_input_width = 224
net_input_height = 224
batch_size = 200


def crop_image(img, skel, intr):
    CUBE_OFFSET = 100  # mm


    def inner_np(depth, skeleton, intr):
        skeleton = np.reshape(skeleton, [21, 3])
        max3d = np.max(skeleton, axis = 0)
        min3d = np.min(skeleton, axis = 0)

        max_z = max3d[2]
        min_z = min3d[2]

        cube = np.array(
                [
                        [min3d[0] - CUBE_OFFSET, min3d[1] - CUBE_OFFSET, min3d[2] - CUBE_OFFSET],  # front top left
                        [max3d[0] + CUBE_OFFSET, min3d[1] - CUBE_OFFSET, min3d[2] - CUBE_OFFSET],  # front top right
                        [max3d[0] + CUBE_OFFSET, max3d[1] + CUBE_OFFSET, min3d[2] - CUBE_OFFSET],  # front bottom right
                        [min3d[0] - CUBE_OFFSET, max3d[1] + CUBE_OFFSET, min3d[2] - CUBE_OFFSET],  # front bottom left

                        [min3d[0] - CUBE_OFFSET, min3d[1] - CUBE_OFFSET, max3d[2] + CUBE_OFFSET],  # back top left
                        [max3d[0] + CUBE_OFFSET, min3d[1] - CUBE_OFFSET, max3d[2] + CUBE_OFFSET],  # back top right
                        [max3d[0] + CUBE_OFFSET, max3d[1] + CUBE_OFFSET, max3d[2] + CUBE_OFFSET],  # back bottom right
                        [min3d[0] - CUBE_OFFSET, max3d[1] + CUBE_OFFSET, max3d[2] + CUBE_OFFSET],  # back bottom right
                        ]
                )

        cube_hom2d = intr.dot(cube.transpose()).transpose()
        cube2d = (cube_hom2d / cube_hom2d[:, 2:])[:, :2]

        max2d = tf.reduce_max(cube2d, axis = 0)
        min2d = tf.reduce_min(cube2d, axis = 0)

        maxx = int(min(net_input_width, max(0, max2d[0])))
        minx = int(min(net_input_width, max(0, min2d[0])))
        maxy = int(min(net_input_height, max(0, max2d[1])))
        miny = int(min(net_input_height, max(0, min2d[1])))

        masked = np.zeros_like(depth, dtype = np.float32)
        masked[miny:maxy, minx:maxx, :] = depth[miny:maxy, minx:maxx, :]

        maxz = max_z + CUBE_OFFSET
        minz = min_z - CUBE_OFFSET

        z_lim = np.logical_or(masked > maxz, masked < minz)
        masked[z_lim] = 0.0

        return masked


    res = tf.numpy_function(inner_np, (img, skel, intr), Tout = tf.float32)
    res = tf.ensure_shape(res, [int(net_input_height), int(net_input_width), 1])
    return res


def prepare_ds(name, ds, cam_intr, crop):
    if crop:
        ds = ds.map(lambda index, depth, img_width, img_height, skeleton, conf_maps:
                    (crop_image(depth, skeleton, cam_intr), conf_maps),
                    num_parallel_calls = tf.data.experimental.AUTOTUNE)
    else:
        ds = ds.map(lambda index, depth, img_width, img_height, skeleton, conf_maps:
                    (depth, conf_maps),
                    num_parallel_calls = tf.data.experimental.AUTOTUNE)

    ds = ds.map(lambda img, confm: (img, decode_confmaps(confm)),
                num_parallel_calls = tf.data.experimental.AUTOTUNE)

    ds = ds.map(lambda img, confm:
                (datasets.util.scale_clip_image_data(img, 1.0 / 1500.0),
                 datasets.util.scale_clip_image_data(confm, 1.0 / 2 ** 16)),
                num_parallel_calls = tf.data.experimental.AUTOTUNE)

    ds = ds.map(lambda img, confm: (img,
                                    confm *
                                    (tf.math.divide_no_nan(
                                            tf.constant(1.0,
                                                        dtype =
                                                        tf.dtypes.float32),
                                            tf.reduce_max(confm)))),
                num_parallel_calls = tf.data.experimental.AUTOTUNE)

    return ds


def batched_twod_argmax(val):
    maxy = tf.argmax(tf.reduce_max(val, axis = 2), 1)
    maxx = tf.argmax(tf.reduce_max(val, axis = 1), 1)
    maxs = tf.stack([maxy, maxx], axis = 2)
    maxs = tf.cast(maxs, dtype = tf.dtypes.float32)
    return maxs


def make_keypoint_metric(index, name, norm_factor):

    def keypoint(y_true, y_pred):
        dist = batched_twod_argmax(y_true) - batched_twod_argmax(y_pred)
        dist = tf.norm(dist, axis = 2)
        mean_dists = norm_factor * tf.reduce_mean(dist, axis = 0)
        return mean_dists[index]


    keypoint.__name__ = 'kp_error_{}'.format(name)
    return keypoint


if __name__ == "__main__":
    with open("datasets.json", "r") as f:
        ds_settings = json.load(f)

    ds_provider = SerializedDataset(ds_settings["NYU224ConfMap"])

    ds_test = ds_provider.get_data("test")
    ds_test = prepare_ds('test',
                         ds_test,
                         cam_intr = ds_provider.camera_intrinsics,
                         crop = False
                         )
    ds_test = ds_test.shuffle(100)

    pose_model_path = 'E:\\Google Drive\\UNI\\Master\\Thesis\\Data\\pose_est\\2d\\noch_weiter_trainiert\\checkpoints\\pose_est_refined.hdf5'
    model = tf.keras.models.load_model(pose_model_path, compile = False)

    import math



    image_diagonal = math.sqrt(net_input_height ** 2 + net_input_width ** 2)
    metric_norm_factor = 1 / image_diagonal

    metrics = [make_keypoint_metric(idx, name, metric_norm_factor) for idx, name in enumerate(ds_provider.joint_names)]

    with open("2d_pose_est_test_results_nyu.csv", "w") as csv:
        headline = ','.join(ds_provider.joint_names)
        csv.write(headline + '\n')
        data = ds_test.take(10000)
        sample = 0
        for x, y in data:
            y_pred = model.predict(tf.expand_dims(x, 0))
            metric_results = []
            for metric in metrics:
                metric_results.append(str(metric(tf.expand_dims(y, 0), y_pred).numpy()))
            csv.write(','.join(metric_results) + '\n')
            sample += 1
            if sample % 100 == 0:
                print(sample)




    # model.compile(optimizer='adam', loss='mse', metrics=metrics)
    # results = model.evaluate(ds_test, verbose=1)
    # with open("2d_pose_est_test_results.txt", "w") as f:
    #     f.write(results)