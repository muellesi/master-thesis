import numpy as np
import os
import tensorflow as tf
from .tfrecord_helper import open_tf_record, decode_img
from .data_downloader import download_data
from google_drive_downloader import GoogleDriveDownloader as gdd

class SerializedDataset:

    def __init__(self, config, cache_path = "C:\\temp"):
        self.name = config["name"]
        self.data_root = config["location"]

        if not os.path.exists(self.data_root):
            if config.get("gdrive_fid") is not None:
                dataset_id = "{}_w{}_h{}".format(self.name, config["depth_width"], config["depth_height"])
                filename = dataset_id + ".zip"
                target_path = os.path.join(cache_path, dataset_id)
                self.data_root = os.path.join(target_path, os.path.basename(self.data_root))
                if not os.path.exists(self.data_root):
                    gdd.download_file_from_google_drive(file_id = config["gdrive_fid"],
                                                        overwrite = True,
                                                        dest_path = os.path.join(target_path, filename),
                                                        showsize = True,
                                                        unzip = True)
            else:
                assert False, "Dataset location has to exist on the system or gdrive_fid has to be set!"

        self.int_pp_x = config["depth_intrinsics"]["pp_x"]
        self.int_pp_y = config["depth_intrinsics"]["pp_y"]
        self.int_f_x = config["depth_intrinsics"]["f_x"]
        self.int_f_y = config["depth_intrinsics"]["f_y"]

        self.camera_intrinsics = np.array([
                [self.int_f_x, 0.0, self.int_pp_x],
                [0.0, self.int_f_y, self.int_pp_y],
                [0.0, 0.0, 1.0]
                ])

        self.joint_names = config["joint_names"]
        self.depth_width = config["depth_width"]
        self.depth_height = config["depth_height"]


    def get_data(self, subset, cache = False):
        ds = tf.data.Dataset.list_files(os.path.join(self.data_root, subset + "*.tfrecord"))
        ds = ds.interleave(open_tf_record, cycle_length = 8, block_length = 1)
        if cache:
            ds = ds.cache()
        ds = ds.map(decode_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        return ds

    def __str__(self):
        return "{}-{}".format(self.name, super().__str__())
