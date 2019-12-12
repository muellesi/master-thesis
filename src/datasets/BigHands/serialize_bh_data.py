""""
    Dataset source: http://icvl.ee.ic.ac.uk/hands17/challenge
    S. Yuan, Q. Ye, G. Garcia-Hernando, T.-K. Kim. The 2017 Hands in the
    Million Challenge on 3D Hand Pose Estimation. arXiv:1707.02237.
    S. Yuan, Q. Ye, B. Stenger, S. Jain, T.-K. Kim. BigHand2. 2M Benchmark:
    Hand Pose Dataset and State of the Art Analysis. CVPR 2017.
    G. Garcia-Hernando, S. Yuan, S. Baek, T.-K. Kim. First-Person Hand
    Action Benchmark with RGB-D Videos and 3D Hand Pose Annotations. CVPR 2018.
    S. Yuan, G. Garcia-Hernando, B. Stenger, et al. Depth-Based 3D Hand Pose
    Estimation: From Current Achievements to Future Goals. CVPR 2018.
"""

import argparse
import os
import shutil

import cv2
import numpy as np
import pandas as pd
import progressbar
import tensorflow as tf

import datasets.tfrecord_helper as tfrh
import datasets.util
import tools



__logger = tools.get_logger(__name__, do_file_logging = False)


def prepare_dataset(dataset_location, image_width, image_height):
    """"
        Serializes the BigHands2.2m dataset at dataset_location to tfrecord
        files at dataset_location/tfrecord_data
    """
    annot_file_path = os.path.join(dataset_location, "Training_Annotation.txt")
    tfrecord_basepath = os.path.join(dataset_location,
                                     "tfrecord_data-{}_{}".format(image_width,
                                                                  image_height))
    img_basepath = os.path.join(dataset_location, "images")

    annot_content = pd.read_csv(annot_file_path, sep = "\t", header = None,
                                usecols = list(range(0, 64)))
    file_names = np.array(annot_content.iloc[:, 0])
    skeleton_annots = np.array(annot_content.iloc[:, 1:])

    subsets = ["train", "validation", "test"]
    subset_starts = [0, 600000, 700000]
    subset_ends = [600000, 700000, 957032]
    max_chunk_size = 5000

    progbar = None

    for s_idx, subset in enumerate(subsets):
        # preparation

        chunk = 0
        print("")
        __logger.info("Serializing subset {} ({} samples)".format(subset,
                                                                  subset_ends[
                                                                      s_idx] -
                                                                  subset_starts[
                                                                      s_idx]))
        if progbar: del progbar
        progbar = progressbar.ProgressBar(
                widgets = [progressbar.Bar(), progressbar.Percentage(),
                           progressbar.ETA()])
        progbar.start(subset_ends[s_idx] - subset_starts[s_idx])

        skip_chunk = False

        for idx in range(subset_starts[s_idx], subset_ends[s_idx]):

            if idx % max_chunk_size == 0 or idx == subset_starts[s_idx]:
                filename = "{}_{}.tfrecord".format(subset, chunk)
                if not os.path.exists(os.path.join(tfrecord_basepath, filename)):
                    record_writer = tf.io.TFRecordWriter(
                            os.path.join(tfrecord_basepath, filename))
                    skip_chunk = False
                else:
                    __logger.warn("File {} already exists. Skipping chunk {}.".format(filename, chunk))
                    skip_chunk = True

                chunk += 1

            if skip_chunk:
                continue

            img_path = os.path.join(img_basepath, file_names[idx])
            skel = skeleton_annots[idx]

            with tf.io.gfile.GFile(img_path, 'rb') as fid:
                img = fid.read()
                if 480 != image_height or 640 != image_width:
                    img_decoded = tf.image.decode_png(img, channels = 1,
                                                      dtype = tf.dtypes.uint16)
                    img_decoded = tf.image.resize(img_decoded,
                                                  [image_height, image_width])
                    img_decoded = tf.cast(img_decoded,
                                          dtype = tf.dtypes.uint16)
                    img_encoded = tf.image.encode_png(img_decoded)
                    img = img_encoded

            cam_intr = np.array([
                    [475.065948, 0.0, 315.944855],
                    [0.0, 475.065857, 245.287079],
                    [0.0, 0.0, 1.0]
                    ])

            skel_2d = tools.project_2d(skel.reshape(21, 3), cam_intr)

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


            tfrecord_str = tfrh.make_standard_pose_record(idx,
                                                          img,
                                                          image_width,
                                                          image_height,
                                                          skel,
                                                          skel_maps_encoded)

            # res = tfrh.parse_standard_pose_record(tfrecord_str)
            # import matplotlib.pyplot as plt
            # fig = plt.figure()
            # depth = np.frombuffer(res[1].numpy(), np.uint8)
            # depth = cv2.imdecode(depth, cv2.IMREAD_UNCHANGED)
            # for img in res[5]:
            #     overlay = img.numpy()
            #     overlay = np.frombuffer(overlay, dtype = np.uint8)
            #     overlay = cv2.imdecode(overlay, cv2.IMREAD_UNCHANGED)
            #     depth = depth + overlay
            #     print(overlay.max())
            # ax = fig.add_subplot(111)
            # ax.imshow(depth)
            # fig.show()

            record_writer.write(tfrecord_str)

            if idx % 10 == 0:
                progbar.update(idx - subset_starts[s_idx])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
            '--root',
            type = str,
            default = 'E:\\MasterDaten\\Datasets\\BigHands2.2m\\',
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
