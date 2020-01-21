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
        one_euro = refiners.OneEuroFilter(freq = 30, mincutoff = 1.0, beta = 0.01)

        do_filter = True
        show_net_output = True
        cam = RealsenseCamera({ 'file': 'realsense_settings.json' })
        run = True
        loop_index = 0
        global_start_time = datetime.now()
        while run:
            loop_index += 1
            loop_start_time = datetime.now()
            time, depth_raw, rgb = cam.get_frame()

            depth = cv2.resize(depth_raw, (224, 224))

            depth = tf.cast(depth, dtype = tf.float32)
            thresh = tf.constant(150, dtype = tf.float32)
            mask = tf.greater(depth, thresh)
            non_zero_depth = tf.boolean_mask(depth, mask)
            closest_distance = tf.reduce_min(non_zero_depth)

            upper_mask = tf.where(tf.less_equal(depth, closest_distance + 500.0),tf.ones_like(depth), tf.zeros_like(depth))
            depth = depth * upper_mask

            depth = datasets.util.scale_clip_image_data(depth, 1.0 / 2500.0)

            depth = np.expand_dims(np.expand_dims(depth, 2), 0)
            res = model.predict(depth)

            coords = twod_argmax(res)
            coords = coords.numpy().squeeze()

            res = res.squeeze()

            values = tf.reduce_max(res, axis = [0, 1]).numpy()

            value_norm = np.linalg.norm(values)
            value_max = np.max(values)
            value_min = np.min(values)

            # print("Norm: {}, Max: {}, Min: {}".format(value_norm, value_max, value_min))

            complete_map = np.sum(res, axis = 2)

            net_img = depth.squeeze()
            if show_net_output:
                net_img = net_img + complete_map
            net_img = tools.colorize_cv(net_img)

            depth_raw = depth_raw.clip(min = None, max = 800)
            prod_img = tools.colorize_cv(depth_raw.squeeze())

            if value_norm > 0.5:

                if do_filter:
                    # coords = lpf.filter(coords)
                    coords = one_euro(coords, (datetime.now() - global_start_time).total_seconds())

                coords_scaled = coords * np.array([480 / 224, 640 / 224])
                tools.render_skeleton(prod_img, np.stack([coords_scaled[:, 1], coords_scaled[:, 0]], axis = 1), True,
                                      np.round(values, 3))

            loop_end_time = datetime.now()
            fps = (1 / (loop_end_time - loop_start_time).microseconds) * 1e6

            cv2.putText(prod_img, "{:.01f} fps".format(fps), (50, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)
            cv2.putText(prod_img, "{:.01f} fps".format(fps), (50, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 1)

            net_img = cv2.cvtColor(net_img, cv2.COLOR_RGB2BGR)
            cv2.imshow(win_name_net, net_img)

            prod_img = cv2.cvtColor(prod_img, cv2.COLOR_RGB2BGR)
            cv2.imshow(win_name_net_prod, prod_img)

            key = cv2.waitKey(1) & 0xFF

            if key == 102:
                do_filter = not do_filter
                print("do_filter: {}".format(do_filter))
            elif key == 27:
                run = False
            elif key == 105: #i
                one_euro.set_beta(one_euro.get_beta() * 10)
                print("Beta: {}".format(one_euro.get_beta()))
            elif key == 107: # k
                one_euro.set_beta(one_euro.get_beta() / 10)
                print("Beta: {}".format(one_euro.get_beta()))
            elif key == 106: # j
                one_euro.set_mincutoff(one_euro.get_mincutoff() / 10)
                print("Min_Cutoff: {}".format(one_euro.get_mincutoff()))
            elif key == 108: # l
                one_euro.set_mincutoff(one_euro.get_mincutoff() * 10)
                print("Min_Cutoff: {}".format(one_euro.get_mincutoff()))
            elif key == 110: # n
                show_net_output = not show_net_output

        del cam
    cv2.destroyAllWindows()
    del model
