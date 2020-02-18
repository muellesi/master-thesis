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
import matplotlib.pyplot as plt

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





    def twod_argmax(val):
        maxy = tf.argmax(tf.reduce_max(val, axis = 2), 1)
        maxx = tf.argmax(tf.reduce_max(val, axis = 1), 1)
        maxs = tf.stack([maxy, maxx], axis = 2)
        maxs = tf.cast(maxs, dtype = tf.dtypes.float32)
        return maxs








    # pose estimation training
    twod_pose_basepath = os.path.join(basepath, "2d_pose")

    if not os.path.exists(twod_pose_basepath):
        os.makedirs(twod_pose_basepath)

        import datasets
        import datasets.util

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
                                [max3d[0] + CUBE_OFFSET, max3d[1] + CUBE_OFFSET, min3d[2] - CUBE_OFFSET],
                                # front bottom right
                                [min3d[0] - CUBE_OFFSET, max3d[1] + CUBE_OFFSET, min3d[2] - CUBE_OFFSET],
                                # front bottom left

                                [min3d[0] - CUBE_OFFSET, min3d[1] - CUBE_OFFSET, max3d[2] + CUBE_OFFSET],  # back top left
                                [max3d[0] + CUBE_OFFSET, min3d[1] - CUBE_OFFSET, max3d[2] + CUBE_OFFSET],  # back top right
                                [max3d[0] + CUBE_OFFSET, max3d[1] + CUBE_OFFSET, max3d[2] + CUBE_OFFSET],
                                # back bottom right
                                [min3d[0] - CUBE_OFFSET, max3d[1] + CUBE_OFFSET, max3d[2] + CUBE_OFFSET],
                                # back bottom right
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


        def prepare_ds(name, ds, cam_intr, add_noise, add_empty, augment):
            ds = ds.map(lambda index, depth, img_width, img_height, skeleton, conf_maps:
                        (crop_image(depth, skeleton, cam_intr), conf_maps),
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

            if add_noise:
                ds = ds.map(lambda img, confm:
                            (datasets.util.add_random_noise(img),
                             confm),
                            num_parallel_calls = tf.data.experimental.AUTOTUNE)

            if augment:
                ds = ds.map(
                        lambda img, confm: datasets.util.augment_depth_and_confmaps(img, confm,
                                                                                    augmentation_probability = 0.6),
                        num_parallel_calls = tf.data.experimental.AUTOTUNE)

                ds = ds.map(lambda img, confm: (img,
                                                confm *
                                                (tf.math.divide_no_nan(
                                                        tf.constant(1.0,
                                                                    dtype =
                                                                    tf.dtypes.float32),
                                                        tf.reduce_max(confm)))),
                            num_parallel_calls = tf.data.experimental.AUTOTUNE)  # restore confmap density after blurring


            return ds



        with open("datasets.json", "r") as f:
            ds_settings = json.load(f)

        ds_provider = SerializedDataset(ds_settings["BigHands224ConfMap"])

        ds_train = ds_provider.get_data("train")
        ds_train = prepare_ds('train',
                              ds_train,
                              cam_intr = ds_provider.camera_intrinsics,
                              add_noise = True,
                              add_empty = False,
                              augment = True)
        ds_train = datasets.util.batch_shuffle_prefetch(ds_train,
                                                        batch_size = batch_size)

        batch = ds_train.take(1)
        data = batch.unbatch()

        model = tf.keras.models.load_model("E:\\Google Drive\\UNI\\Master\\Thesis\\Data\\pose_est\\2d\\noch_weiter_trainiert\\checkpoints\\pose_est_refined.hdf5", compile = False)

        idx = 0
        for inp, outp in data:
            pred_res = model.predict(tf.expand_dims(inp, 0))
            skeleton_coords = twod_argmax(pred_res).numpy().squeeze()

            pred_res = pred_res.squeeze()

            input_image = tools.colorize_cv(copy.copy(inp.numpy()), cmap = 'bone')
            cv2.imwrite(os.path.join(twod_pose_basepath, "{}_input.png".format(idx)), input_image)

            confmaps = copy.deepcopy(pred_res)
            confmaps = confmaps.transpose((2,0,1))

            j_idx = 0
            for confmap in confmaps:
                cm = tools.colorize_cv(copy.copy(confmap), cmap = 'bone')
                cv2.imwrite(os.path.join(twod_pose_basepath, "{}_joint_{}_({}).png".format(idx, j_idx, ds_provider.joint_names[j_idx])), cm)
                j_idx += 1

            confmaps_stacked = np.max(confmaps, axis = 0)
            confm_stack_colored = tools.colorize_cv(copy.copy(confmaps_stacked), cmap='bone')
            cv2.imwrite(os.path.join(twod_pose_basepath, "{}_confmap_stack.png".format(idx)), confm_stack_colored)

            skeleton_coords[:] = skeleton_coords[:, [1, 0]]  # ...?
            empty = np.zeros_like(input_image)
            skeleton_img = tools.render_skeleton(empty, skeleton_coords, joint_idxs = False)

            plt.matshow(skeleton_img)
            cv2.imwrite(os.path.join(twod_pose_basepath, "{}_skeleton.png".format(idx)), skeleton_img)

            skeleton_on_depth = tools.render_skeleton(input_image, skeleton_coords, joint_idxs = False)
            cv2.imwrite(os.path.join(twod_pose_basepath, "{}_skeleton_on_img.png".format(idx)),
                        skeleton_on_depth)

            idx += 1











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

