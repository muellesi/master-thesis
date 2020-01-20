import json
from datetime import datetime

import cv2
import numpy as np
import tensorflow as tf

import datasets
import datasets.util
import refiners
import tools
from datasets import SerializedDataset
from datasets.tfrecord_helper import depth_and_confmaps
from tools import RealsenseCamera



use_camera = True


def prepare_ds(ds, add_noise):
    ds = ds.map(depth_and_confmaps,
                num_parallel_calls = tf.data.experimental.AUTOTUNE)
    ds = ds.map(lambda img, confm:
                datasets.util.scale_clip_image_data(img, 1.0 / 2500.0),
                num_parallel_calls = tf.data.experimental.AUTOTUNE)

    if add_noise:
        ds = ds.map(lambda img: datasets.util.add_random_noise(img),
                    num_parallel_calls = tf.data.experimental.AUTOTUNE)

    return ds


def twod_argmax(val):
    maxy = tf.argmax(tf.reduce_max(val, axis = 2), 1)
    maxx = tf.argmax(tf.reduce_max(val, axis = 1), 1)
    maxs = tf.stack([maxy, maxx], axis = 2)
    maxs = tf.cast(maxs, dtype = tf.dtypes.float32)
    return maxs


if __name__ == '__main__':

    model = tf.keras.models.load_model(
            'E:\\Google Drive\\UNI\\Master\\Thesis\\Data\\pose_est\\2d'
            '\\ueber_weihnachten\\pose_est_refined.hdf5', compile = False)

    win_name_net = 'net'
    win_name_net_prod = 'prod'
    cv2.namedWindow(win_name_net)
    cv2.namedWindow(win_name_net_prod)

    if not use_camera:
        with open("datasets.json", "r") as f:
            ds_settings = json.load(f)

        ds_provider = SerializedDataset(ds_settings["BigHands224ConfMap"])
        ds = ds_provider.get_data("test")
        ds = prepare_ds(ds, add_noise = True)
        for img in ds:
            res = model.predict(np.expand_dims(img, 0))
            res = res.squeeze()
            complete_map = np.sum(res, axis = 2)
            img2 = tools.colorize_cv(img.numpy().squeeze() + complete_map)
            cv2.imshow(win_name_net, img2)
            cv2.waitKey(33)
    else:
        lpf = refiners.LowpassRefiner(2, 30, max_sample_history = 300)
        one_euro = refiners.OneEuroFilter(data_shape = (21, 2), min_cutoff = 1.0, beta = 1.0, d_cutoff = 1.0)

        do_filter = False
        cam = RealsenseCamera({ 'file': 'realsense_settings.json' })
        run = True
        loop_index = 0
        while run:
            loop_index += 1
            loop_start_time = datetime.now()
            time, depth_raw, rgb = cam.get_frame()
            depth = cv2.resize(depth_raw, (224, 224))
            depth = datasets.util.scale_clip_image_data(depth, 1.0 / 1000.0)
            depth = np.expand_dims(np.expand_dims(depth, 2), 0)
            res = model.predict(depth)
            coords = twod_argmax(res)
            coords = coords.numpy().squeeze()

            if do_filter:
                # coords = lpf.filter(coords)
                coords = one_euro(loop_index, coords)

            res = res.squeeze()
            values = tf.reduce_max(res, axis = [0, 1]).numpy()

            value_norm = np.linalg.norm(values)
            value_max = np.max(values)
            value_min = np.min(values)
            # print("Norm: {}, Max: {}, Min: {}".format(value_norm, value_max, value_min))

            complete_map = np.sum(res, axis = 2)
            net_img = depth.squeeze() + complete_map
            net_img = tools.colorize_cv(net_img)

            depth_raw = depth_raw.clip(min = None, max = 800)

            prod_img = tools.colorize_cv(depth_raw.squeeze())

            if value_norm > 0.5:
                coords_scaled = coords * np.array([480 / 224, 640 / 224])
                tools.render_skeleton(prod_img, np.stack([coords_scaled[:, 1], coords_scaled[:, 0]], axis = 1), True,
                                      np.round(values, 3))

                for coord, value in zip(coords_scaled, values):
                    prod_img = cv2.circle(prod_img, (int(coord[1]), int(coord[0])), 3, (0, 0, 0))

            loop_end_time = datetime.now()
            fps = (1 / (loop_end_time - loop_start_time).microseconds) * 1e6

            cv2.putText(prod_img, "{:.01f} fps".format(fps), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0))

            net_img = cv2.cvtColor(net_img, cv2.COLOR_RGB2BGR)
            cv2.imshow(win_name_net, net_img)

            prod_img = cv2.cvtColor(prod_img, cv2.COLOR_RGB2BGR)
            cv2.imshow(win_name_net_prod, prod_img)

            key = cv2.waitKey(1) & 0xFF

            if key == 102:
                do_filter = not do_filter
                print("do_filter: {}".format(do_filter))
            if key == 27:
                run = False

        del cam
    cv2.destroyAllWindows()
    del model
