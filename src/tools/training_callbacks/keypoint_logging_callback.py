import os

import tensorflow as tf



def batched_twod_argmax(val):
    maxy = tf.argmax(tf.reduce_max(val, axis = 2), 1)
    maxx = tf.argmax(tf.reduce_max(val, axis = 1), 1)
    maxs = tf.stack([maxy, maxx], axis = 2)
    maxs = tf.cast(maxs, dtype = tf.dtypes.float32)
    return maxs


def make_keypoint_metric(index, name):
    def keypoint(y_true, y_pred):
        dist = batched_twod_argmax(y_true) - batched_twod_argmax(y_pred)
        dist = tf.norm(dist, axis = 2)
        mean_dists = tf.reduce_mean(dist, axis = 0)
        return mean_dists[index]
    keypoint.__name__ = 'kp_error_{}'.format(name)
    return keypoint


class KeypointLoggingCallback(tf.keras.callbacks.Callback):

    def __init__(self, log_dir, keypoint_names):
        super(KeypointLoggingCallback, self).__init__()

        self.logdir = os.path.join(log_dir, "keypoint_metrics")
        os.makedirs(self.logdir, exist_ok = True)

        self.writer = tf.summary.create_file_writer(self.logdir)
        self.keypoint_names = keypoint_names
        self.num_keypoints = len(self.keypoint_names)

        self.metric_prefix = "kp_error_"


    def on_batch_end(self, batch, logs = None):
        with self.writer.as_default():
            for i in range(self.num_keypoints):
                metric_name = self.metric_prefix + str(i)
                metric_value = logs.get(metric_name, 0)
                tf.summary.scalar("mean_error_" + self.keypoint_names[i], metric_value, batch)


    def generate_metrics(self):
        return [make_keypoint_metric(i, self.keypoint_names[i]) for i in range(self.num_keypoints)]
