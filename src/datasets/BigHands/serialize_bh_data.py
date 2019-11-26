""""
    Dataset source: http://icvl.ee.ic.ac.uk/hands17/challenge
    S. Yuan, Q. Ye, G. Garcia-Hernando, T.-K. Kim. The 2017 Hands in the Million Challenge on 3D Hand Pose Estimation. arXiv:1707.02237.
    S. Yuan, Q. Ye, B. Stenger, S. Jain, T.-K. Kim. BigHand2. 2M Benchmark: Hand Pose Dataset and State of the Art Analysis. CVPR 2017.
    G. Garcia-Hernando, S. Yuan, S. Baek, T.-K. Kim. First-Person Hand Action Benchmark with RGB-D Videos and 3D Hand Pose Annotations. CVPR 2018.
    S. Yuan, G. Garcia-Hernando, B. Stenger, et al. Depth-Based 3D Hand Pose Estimation: From Current Achievements to Future Goals. CVPR 2018.
"""

import tensorflow as tf
import os
import numpy as np
import shutil
import progressbar
import tools
import datasets.tfrecord_helper as tfrh
import pandas as pd
import argparse



__logger = tools.get_logger(__name__, do_file_logging=False)


def prepare_dataset(dataset_location):
    """"
        Serializes the BigHands2.2m dataset at dataset_location to tfrecord files at dataset_location/tfrecord_data
    """
    annot_file_path = os.path.join(dataset_location, "Training_Annotation.txt")
    tfrecord_basepath = os.path.join(dataset_location, "tfrecord_data")
    img_basepath = os.path.join(dataset_location, "images")

    if os.path.exists(tfrecord_basepath):
        shutil.rmtree(tfrecord_basepath)

    os.makedirs(tfrecord_basepath)

    annot_content = pd.read_csv(annot_file_path, sep="\t", header=None, usecols=list(range(0, 64)))
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
        __logger.info("Serializing subset {} ({} samples)".format(subset, subset_ends[s_idx] - subset_starts[s_idx]))
        if progbar: del progbar
        progbar = progressbar.ProgressBar(widgets=[progressbar.Bar(), progressbar.Percentage(), progressbar.ETA()])
        progbar.start(subset_ends[s_idx] - subset_starts[s_idx])

        for idx in range(subset_starts[s_idx], subset_ends[s_idx]):
            img_path = os.path.join(img_basepath, file_names[idx])
            skel = skeleton_annots[idx]

            with tf.io.gfile.GFile(img_path, 'rb') as fid:
                img = fid.read()

            if idx % max_chunk_size == 0:
                filename = "{}_{}.tfrecord".format(subset, chunk)
                record_writer = tf.io.TFRecordWriter(os.path.join(tfrecord_basepath, filename))
                chunk += 1

            tfrecord_str = tfrh.make_standard_pose_record(idx, img, 640, 480, skel)
            record_writer.write(tfrecord_str)

            if idx % 10 == 0:
                progbar.update(idx - subset_starts[s_idx])


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
