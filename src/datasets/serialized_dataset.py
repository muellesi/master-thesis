import numpy as np
import os
import tensorflow as tf
from .tfrecord_helper import open_tf_record, decode_img



class SerializedDataset:

    def __init__(self, config):
        self.name = config["name"]

        self.data_root = config["location"]
        assert (os.path.exists(self.data_root))

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


    def get_data(self, subset):
        filenames = np.array([os.path.abspath(os.path.join(self.data_root, f))
                              for f in os.listdir(self.data_root) if f.startswith(subset)])
        ds = tf.data.Dataset.from_tensor_slices(filenames).cache()
        ds = ds.flat_map(open_tf_record)
        ds = ds.map(decode_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        return ds

    def __str__(self):
        return "{}-{}".format(self.name, super().__str__())
