import tensorflow as tf
import os
import numpy as np
import argparse
import shutil
import cv2
import progressbar
import tools
import datasets.tfrecord_helper as tfrh
import pathlib
import struct



__logger = tools.get_logger(__name__, do_file_logging=False)


def reconstruct_image(binfile):
    """
    Quote from the readme:
    While the depth image is 320x240, the valid hand region is usually much smaller. To save space, each *.bin file only stores the bounding
    box of the hand region. Specifically, each bin file starts with 6 unsigned int: img_width img_height left top right bottom. [left, right) and [top, bottom)
    is the bounding box coordinate in this depth image. The bin file then stores all the depth pixel values in the bounding box in row scanning order,
    which are  (right - left) * (bottom - top) floats. The unit is millimeters. The bin file is binary and needs to be opened with std::ios::binary flag.
    """

    with open(binfile, 'rb') as f:
        binary = f.read()
    img_width, img_height, left, top, right, bottom = struct.unpack("IIIIII", binary[:6 * 4])
    cropped_image = np.frombuffer(binary[6 * 4:], dtype=np.float32)
    cropped_image = cropped_image.reshape(((bottom - top), (right - left)))
    new_image = np.pad(cropped_image, [[top, img_height - bottom], [left, img_width - right]], constant_values=0)
    return new_image


def load_and_reencode(binfile):
    raw_image = reconstruct_image(binfile)
    raw_image = cv2.resize(raw_image, (640, 480))
    success, encoded = cv2.imencode(".png", raw_image.astype(np.uint16))
    return encoded.tobytes()


def prepare_dataset(dataset_location):
    tfrecord_basepath = os.path.join(dataset_location, "tfrecord_data")

    train_subjects = ["P0", "P2", "P3", "P4", "P5", "P6"]
    validate_subjects = ["P1"]

    skeleton_files = [file for file in pathlib.Path(dataset_location).rglob("joint.txt")]

    if os.path.exists(tfrecord_basepath):
        shutil.rmtree(tfrecord_basepath)

    os.makedirs(tfrecord_basepath)

    train_skels = []
    validation_skels = []
    test_skels = []
    for skeleton_file in skeleton_files:
        subject = skeleton_file.relative_to(dataset_location).parts[0]
        if subject in train_subjects:
            train_skels.append(skeleton_file)
        elif subject in validate_subjects:
            validation_skels.append(skeleton_file)
        else:
            test_skels.append(skeleton_file)

    all_skel_files = np.array([train_skels, validation_skels, test_skels])

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
            with open(file, 'r') as joint_f:
                num_frames = int(joint_f.readline().strip())

            skels = np.loadtxt(file, skiprows=1)

            current_directory = file.parent

            for frame_index in range(num_frames):
                skeleton = skels[frame_index]

                # map to fhad joint ordering:
                # msra readme: The 21 hand joints are: wrist, index_mcp, index_pip, index_dip, index_tip, middle_mcp, middle_pip, middle_dip, middle_tip, ring_mcp, ring_pip, ring_dip, ring_tip, little_mcp, little_pip, little_dip, little_tip, thumb_mcp, thumb_pip, thumb_dip, thumb_tip.
                # this is reordered to: ["wrist", "tmcp", "imcp", "mmcp", "rmcp", "pmcp", "tpip", "tdip", "ttip", "ipip", "idip", "itip", "mpip", "mdip", "mtip", "rpip", "rdip", "rtip", "ppip", "pdip", "ptip"]
                skeleton = skeleton.reshape((21, -1))
                skeleton = skeleton[[0, 17, 1, 5, 9, 13, 18, 19, 20, 2, 3, 4, 6, 7, 8, 10, 11, 12, 14, 15, 16], :]
                skeleton = skeleton.dot(np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]]))
                skeleton = skeleton.reshape((-1))
                img_filename = "{:06d}_depth.bin".format(frame_index)

                img_path = os.path.join(current_directory, img_filename)

                img = load_and_reencode(img_path)

                if sample_idx % max_chunk_size == 0:
                    # write data to chunk file
                    filename = "{}_{}.tfrecord".format(subset, chunk_idx)
                    record_writer = tf.io.TFRecordWriter(os.path.join(tfrecord_basepath, filename))
                    chunk_idx += 1

                tfrecord_str = tfrh.make_standard_pose_record(sample_idx, img, 640, 480, skeleton)
                record_writer.write(tfrecord_str)
                sample_idx += 1

            progbar.update(file_idx)


@tf.function
def __decode_img(index, depth, img_width, img_height, skeleton):
    img = tf.image.decode_png(depth, channels=1, dtype=tf.uint16)
    return img, skeleton


@tf.function
def __open_tf_record(filename):
    tfrecord = tf.data.TFRecordDataset(filename)
    return tfrecord.map(tfrh.parse_standard_pose_record)


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
