import tensorflow as tf


@tf.function
def flip_randomly(img, skel):
    rand = tf.random.uniform(())

    if rand > 0.5:
        img = tf.image.flip_left_right(img)
        skel = tf.reshape(skel, [21, 3])
        skel = tf.matmul(skel, tf.constant([-1, 0, 0, 0, 1, 0, 0, 0, 1], shape=[3, 3], dtype=tf.float32))
        skel = tf.reshape(skel, [63])

    return img, skel