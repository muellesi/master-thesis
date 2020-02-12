""""
    Dataset source: https://guiggh.github.io/publications/first-person-hands/
    "First-Person Hand Action Benchmark with RGB-D Videos and 3D Hand Pose Annotations"
    Garcia-Hernando, Guillermo and Yuan, Shanxin and Baek, Seungryul and Kim, Tae-Kyun.
    Proceedings of Computer Vision and Pattern Recognition ({CVPR}), 2018
"""

import argparse
import os
import pathlib
import shutil

import cv2
import numpy as np
import progressbar
import tensorflow as tf

import datasets.tfrecord_helper as tfrh
import datasets.util
import tools

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)

__logger = tools.get_logger(__name__, do_file_logging = False)


def prepare_dataset(dataset_location, image_width, image_height):
    """"
        Serializes the FHAD dataset at dataset_location to tfrecord files at dataset_location/tfrecord_data
    """
    skeleton_location = os.path.join(dataset_location, "Hand_pose_annotation_v1")
    img_basepath = os.path.join(dataset_location, "Video_files")
    tfrecord_basepath = os.path.join(dataset_location,
                                     "tfrecord_data-{}_{}".format(image_width,
                                                                  image_height))
    train_subjects = ["Subject_1", "Subject_3", "Subject_4"]
    validate_subjects = ["Subject_2"]

    skeleton_files = [file for file in pathlib.Path(skeleton_location).rglob("skeleton.txt")]

    if os.path.exists(tfrecord_basepath):
        shutil.rmtree(tfrecord_basepath)

    os.makedirs(tfrecord_basepath)

    train_skel_files = []
    val_skel_files = []
    test_skel_files = []
    for skeleton_file in skeleton_files:
        relative_path = pathlib.Path(os.path.relpath(str(skeleton_file.absolute()), skeleton_location))
        subject = relative_path.parts[0]
        if subject in train_subjects:
            train_skel_files.append(skeleton_file)
        elif subject in validate_subjects:
            val_skel_files.append(skeleton_file)
        else:
            test_skel_files.append(skeleton_file)

    all_skel_files = np.array([train_skel_files, val_skel_files, test_skel_files])

    subsets = ["train", "validation", "test"]
    max_chunk_size = 5000

    progbar = None

    for idx, subset in enumerate(subsets):
        __logger.info("Creating serialized files for {} set".format(subset))
        sample_idx = 0
        chunk_idx = 0

        if progbar: del progbar
        progbar = progressbar.ProgressBar(widgets = [progressbar.Bar(), progressbar.Percentage(), progressbar.ETA()])
        progbar.start(max_value = len(all_skel_files[idx]))

        for file_idx, file in enumerate(all_skel_files[idx]):

            skels_world_coords = np.loadtxt(file)
            specific_img_basepath = os.path.join(img_basepath, file.relative_to(skeleton_location).parent, "depth")

            for line in skels_world_coords:
                frame_index = int(line[0])

                skeleton = line[1:]
                skeleton = skeleton.reshape((21, -1))
                skeleton = tools.skeleton_renderer.project_world_to_cam(skeleton, get_camera_extrinsics())
                skeleton = skeleton.reshape((-1))

                img_filename = "depth_{:04d}.png".format(frame_index)
                img_path = os.path.join(specific_img_basepath, img_filename)

                with tf.io.gfile.GFile(img_path, 'rb') as fid:
                    img = fid.read()
                    if 480 != image_height or 640 != image_width:
                        img_decoded = tf.image.decode_png(img, channels = 1, dtype = tf.dtypes.uint16)
                        img_decoded = tf.image.resize(img_decoded, [image_height, image_width])
                        img_decoded = tf.cast(img_decoded, dtype = tf.dtypes.uint16)
                        img_encoded = tf.image.encode_png(img_decoded)
                        img = img_encoded

                if sample_idx % max_chunk_size == 0:
                    # write data to chunk file
                    filename = "{}_{}.tfrecord".format(subset, chunk_idx)
                    record_writer = tf.io.TFRecordWriter(os.path.join(tfrecord_basepath, filename))
                    chunk_idx += 1

                cam_intr = np.array([
                        [475.065948, 0.0, 315.944855],
                        [0.0, 475.065857, 245.287079],
                        [0.0, 0.0, 1.0]
                        ])

                skel_2d = tools.project_2d(skeleton.reshape(21, 3), cam_intr)

                if 480 != image_height or 640 != image_width:
                    skel_2d = skel_2d.dot(np.array([[image_width / 640.0, 0.0],
                                                    [0.0, image_height / 480.0]]))

                skel_maps = datasets.util.pts_to_confmaps(skel_2d,
                                                          image_width,
                                                          image_height,
                                                          sigma = 10)
                skel_maps = skel_maps * 2 ** 16
                skel_maps = skel_maps.astype(np.uint16)
                skel_maps_encoded = [
                        cv2.imencode(".png", m[:, :, np.newaxis])[1].tostring() for
                        m in skel_maps]

                tfrecord_str = tfrh.make_standard_pose_record_with_confmaps(sample_idx,
                                                                            img,
                                                                            image_width,
                                                                            image_height,
                                                                            skeleton,
                                                                            skel_maps_encoded)
                record_writer.write(tfrecord_str)
                sample_idx += 1

            progbar.update(file_idx)


def get_camera_extrinsics():
    """
    Returns the camera's extrinsic matrix
    source: https://github.com/guiggh/hand_pose_action/blob/master/load_example.py#L147
    :return: camera extrinsics
    """
    cam_extr = np.array(
            [[0.999988496304, -0.00468848412856, 0.000982563360594, 25.7],
             [0.00469115935266, 0.999985218048, -0.00273845880292, 1.22],
             [-0.000969709653873, 0.00274303671904, 0.99999576807, 3.902],
             [0, 0, 0, 1.0]])
    return cam_extr


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
            '--root',
            type = str,
            required = True,
            help = 'Root path for the dataset'
            )
    parser.add_argument(
            '--image_width',
            type = str,
            default = 224,
            help = 'Width of input images and skeleton maps'
            )
    parser.add_argument(
            '--image_height',
            type = str,
            default = 224,
            help = 'Height of input images and skeleton maps'
            )
    args = parser.parse_args()

    prepare_dataset(args.root, args.image_width, args.image_height)
