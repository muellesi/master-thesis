import copy
import json
import os
import re

import cv2
import numpy as np
import tensorflow as tf

import tools.training_callbacks
from datasets import SerializedDataset
from datasets.tfrecord_helper import decode_confmaps
import glob


net_input_height = 224
net_input_width = 224
batch_size = 20


if __name__ == "__main__":
    basepath = "dataset_renders"


    def get_valid_filename(s):
        s = str(s).strip().replace(' ', '_')
        return re.sub(r'(?u)[^-\w.]', '', s)


    with open("datasets.json", "r") as f:
        ds_settings = json.load(f)

    for key in ds_settings.keys():

        if "colab" in key.lower():
            continue

        ds_render_basepath = os.path.join(basepath, key)
        if os.path.exists(ds_render_basepath):
            continue

        os.makedirs(ds_render_basepath, exist_ok = True)

        try:
            ds_provider = SerializedDataset(ds_settings[key])
        except:
            continue

        ds_train = ds_provider.get_data("train")
        if ds_provider.has_confmaps:
            ds_train = ds_train.map(lambda index, img, img_width, img_height, skeleton, conf_maps: (
                    img, decode_confmaps(conf_maps), skeleton))
        else:
            ds_train = ds_train.map(lambda index, img, img_width, img_height, skeleton: (img,
                                                                                         tf.stack([tf.zeros_like(img)] *
                                                                                                  tf.shape(tf.reshape(
                                                                                                          skeleton,
                                                                                                          [-1, 3]))[0]),
                                                                                         skeleton))

        ds_train = ds_train.batch(36)

        batch = ds_train.take(1)
        data = batch.unbatch()

        sample_index = 0
        for depth, confm, skel in data:

            depth = np.squeeze(depth)

            skel = np.array(skel).reshape((-1, 3))

            skel = tools.project_2d(skel, ds_provider.camera_intrinsics)
            if key == 'MSRA':
                skel = skel * 2.0

            depth_colored_viridis = tools.colorize_cv(copy.copy(depth), cmap = "viridis")
            depth_colored_bone = tools.colorize_cv(copy.copy(depth), cmap = 'bone')

            # plot image only

            cv2.imwrite(os.path.join(ds_render_basepath, "{}_input_image_viridis.png".format(sample_index)),
                        depth_colored_viridis)
            cv2.imwrite(os.path.join(ds_render_basepath, "{}_input_image_jet.png".format(sample_index)),
                        depth_colored_bone)

            # plot confmaps extra
            if ds_provider.has_confmaps:
                confm = np.squeeze(confm)
                confm = np.array(confm).transpose([2, 0, 1])

                confm_idx = 0
                for confmap in confm:
                    confmap_colored = tools.colorize_cv(copy.copy(confmap), cmap = "viridis")
                    cv2.imwrite(os.path.join(ds_render_basepath, "{}_confmap_{}({}).png".format(sample_index, confm_idx,
                                                                                                ds_provider.joint_names[
                                                                                                    confm_idx])),
                                confmap_colored)
                    confm_idx += 1

                # plot confmaps stacked
                stacked = copy.copy(depth)
                stacked += np.max(confm,
                                  axis = 0)  # get fake 'stack' of confidence maps that does not bleed out too much if multiple peaks overlap
                stacked = tools.colorize_cv(stacked, cmap = 'viridis')
                cv2.imwrite(os.path.join(ds_render_basepath, "{}_confmaps_stacked.png".format(sample_index)), stacked)

            # plot skeleton
            empty = np.zeros_like(depth_colored_viridis)
            skeleton_img = tools.render_skeleton(empty, skel, joint_idxs = False)
            cv2.imwrite(os.path.join(ds_render_basepath, "{}_skeleton.png".format(sample_index)), skeleton_img)

            skeleton_on_depth = tools.render_skeleton(copy.copy(depth_colored_bone), skel, joint_idxs = False)
            cv2.imwrite(os.path.join(ds_render_basepath, "{}_skeleton_on_img.png".format(sample_index)),
                        skeleton_on_depth)

            sample_index += 1

        del ds_train
        del ds_provider



    # Structural training dataset



    def get_filepaths(filepaths):
        for file in filepaths:
            yield file


    def decode_img(raw):
        img = tf.image.decode_png(raw, channels = 1, dtype = tf.dtypes.uint16)
        return img


    def load_image(path):
        img = tf.io.read_file(path)
        img = decode_img(img)
        return img


    def scale_image(img):
        img = tf.cast(tf.image.resize(img, tf.constant([net_input_height, net_input_width], dtype = tf.dtypes.int32)),
                      dtype = tf.float32)
        img = img / tf.constant(2500.0, dtype = tf.float32)
        img = tf.clip_by_value(img, clip_value_min = 0.0, clip_value_max = 1.0)  # ignore stuff more than 2.5m away.
        return img


    def duplicate_image(img):
        return img, img  # for autoencoder - x == y


    def batch_shuffle_prefetch(ds):
        # ds = ds.repeat()
        ds = ds.shuffle(batch_size * 20)
        ds = ds.batch(batch_size = batch_size)
        ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
        return ds


    def add_random_noise(img):
        noise = tf.random.normal(shape = tf.shape(img), mean = 0.0, stddev = 0.01, dtype = tf.dtypes.float32)
        return img + noise


    def add_random_bg_image(img, bg_images):
        bg = next(iter(bg_images))
        # For this to work we need to remove every pixel that
        # is NOT zero in the 'hand image' from the background
        # before adding both images
        mask = tf.math.sign(img)  # mask hand with 1, bg with 0. Everything that is now 1 has to be removed from bg
        cond = tf.math.equal(mask, tf.ones(tf.shape(mask)))  # convert to bool array
        mask = tf.where(cond, tf.zeros(tf.shape(mask)), tf.ones(
            tf.shape(mask)))  # use bool array to 'invert' mask -> former 1s are now 0s and vice versa
        return bg * mask + img


    def prepare(ds, add_noise, bg_images = None):
        ds = ds.map(load_image, num_parallel_calls = tf.data.experimental.AUTOTUNE)
        ds = ds.map(scale_image, num_parallel_calls = tf.data.experimental.AUTOTUNE).cache()
        ds = ds.map(duplicate_image, num_parallel_calls = tf.data.experimental.AUTOTUNE)
        if bg_images:
            ds = ds.map(lambda img1, img2: (add_random_bg_image(img1, bg_images), img2))
        if add_noise:
            ds = ds.map(lambda img1, img2: (add_random_noise(img1), img2),
                        num_parallel_calls = tf.data.experimental.AUTOTUNE)

        return batch_shuffle_prefetch(ds)


    def get_data(data_root, val_split = 0.33):
        all_files = glob.glob(os.path.join(data_root, "hands", "**", "*.png"), recursive = True)
        all_files = np.array(all_files)
        np.random.shuffle(all_files)

        val_last_index = int(round(val_split * len(all_files)))
        files_val = all_files[:val_last_index]
        files_train = all_files[val_last_index:]

        augment_backgrounds = tf.data.Dataset.list_files(os.path.join(data_root, "augmentation", "**", "*.png"),
                                                         shuffle = True)
        augment_backgrounds = augment_backgrounds.map(load_image, num_parallel_calls = tf.data.experimental.AUTOTUNE)
        augment_backgrounds = augment_backgrounds.map(scale_image,
                                                      num_parallel_calls = tf.data.experimental.AUTOTUNE).cache()

        print("Iterating backgrounds once to cache all images...")
        for img in augment_backgrounds:  # cache all images...
            pass
        print("done!")
        augment_backgrounds = augment_backgrounds.repeat().shuffle(5).prefetch(tf.data.experimental.AUTOTUNE)

        ds_train = tf.data.Dataset.from_tensor_slices(np.array(files_train)).cache()
        ds_train = prepare(ds_train, add_noise = True, bg_images = augment_backgrounds)

        ds_val = tf.data.Dataset.from_tensor_slices(np.array(files_val)).cache()
        ds_val = prepare(ds_val, add_noise = True, bg_images = augment_backgrounds)

        return ds_train, ds_val

    structural_basepath = os.path.join(basepath, "structural")

    if not os.path.exists(structural_basepath):
        os.makedirs(structural_basepath, exist_ok = True)

        ds_train, ds_val = get_data("E:\\MasterDaten\\Datasets\\StructuralLearning")
        batch = ds_train.take(1)
        data = batch.unbatch()

        model = tf.keras.models.load_model("E:\\Google Drive\\UNI\\Master\\Thesis\\Data\\structural\\ae_with_bg_high_acc\\full_ae_weights.epoch_47.hdf5", compile = False)
        predict_result = model.predict(batch)

        idx = 0
        for inp, outp in data:
            pred_res = model.predict(tf.expand_dims(inp, 0)).squeeze()
            input = np.array(inp).squeeze()
            output = np.array(outp).squeeze()

            augmented = tools.colorize_cv(input, cmap = 'bone')
            normal = tools.colorize_cv(output, cmap = 'bone')
            pred = tools.colorize_cv(pred_res, cmap = 'bone')

            cv2.imwrite(os.path.join(structural_basepath, "{}_augmented.png".format(idx)), augmented)
            cv2.imwrite(os.path.join(structural_basepath, "{}_normal.png".format(idx)), normal)
            cv2.imwrite(os.path.join(structural_basepath, "{}_predict.png".format(idx)), pred)

            idx += 1

