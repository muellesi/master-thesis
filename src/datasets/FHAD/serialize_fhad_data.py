""""
    Dataset source: https://guiggh.github.io/publications/first-person-hands/
    "First-Person Hand Action Benchmark with RGB-D Videos and 3D Hand Pose Annotations"
    Garcia-Hernando, Guillermo and Yuan, Shanxin and Baek, Seungryul and Kim, Tae-Kyun.
    Proceedings of Computer Vision and Pattern Recognition ({CVPR}), 2018
"""

import tensorflow as tf
import os
import numpy as np
import shutil
import progressbar
import tools
import datasets.tfrecord_helper as tfrh
import pathlib
import argparse



__logger = tools.get_logger(__name__, do_file_logging=False)


def prepare_dataset(dataset_location):
    """"
        Serializes the FHAD dataset at dataset_location to tfrecord files at dataset_location/tfrecord_data
    """
    skeleton_location = os.path.join(dataset_location, "Hand_pose_annotation_v1")
    img_basepath = os.path.join(dataset_location, "Video_files")
    tfrecord_basepath = os.path.join(dataset_location, "tfrecord_data")

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
        progbar = progressbar.ProgressBar(widgets=[progressbar.Bar(), progressbar.Percentage(), progressbar.ETA()])
        progbar.start(max_value=len(all_skel_files[idx]))

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

                if sample_idx % max_chunk_size == 0:
                    # write data to chunk file
                    filename = "{}_{}.tfrecord".format(subset, chunk_idx)
                    record_writer = tf.io.TFRecordWriter(os.path.join(tfrecord_basepath, filename))
                    chunk_idx += 1

                tfrecord_str = tfrh.make_standard_pose_record(sample_idx, img, 640, 480, skeleton)
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
            type=str,
            required=True,
            help='Root path for the dataset'
            )
    args = parser.parse_args()

    prepare_dataset(args.root)
