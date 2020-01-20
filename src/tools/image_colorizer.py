import matplotlib
import matplotlib.cm

import tensorflow as tf



def colorize_tf(value, vmin = None, vmax = None, cmap = None):
    """
    Source: https://gist.github.com/jimfleming/c1adfdb0f526465c99409cc143dea97b

    A utility function for TensorFlow that maps a grayscale image to a matplotlib
    colormap for use with TensorBoard image summaries.

    By default it will normalize the input value to the range 0..1 before mapping
    to a grayscale colormap.

    Arguments:
      - value: 2D Tensor of shape [height, width] or 3D Tensor of shape
        [height, width, 1].
      - vmin: the minimum value of the range used for normalization.
        (Default: value minimum)
      - vmax: the maximum value of the range used for normalization.
        (Default: value maximum)
      - cmap: a valid cmap named for use with matplotlib's `get_cmap`.
        (Default: 'gray')

    Example usage:

    ```
    output = tf.random_uniform(shape=[256, 256, 1])
    output_color = colorize(output, vmin=0.0, vmax=1.0, cmap='viridis')
    tf.summary.image('output', output_color)
    ```

    Returns a 3D tensor of shape [height, width, 3].
    """

    # normalize
    vmin = tf.reduce_min(value) if vmin is None else vmin
    vmax = tf.reduce_max(value) if vmax is None else vmax
    value = (value - vmin) / (vmax - vmin)  # vmin..vmax

    # squeeze last dim if it exists
    value = tf.squeeze(value)

    # quantize
    indices = tf.cast(tf.round(value * 255), tf.int32)

    # gather
    cm = matplotlib.cm.get_cmap(cmap if cmap is not None else 'viridis')
    colors = tf.constant([cm(i) for i in range(256)], dtype = tf.float32)
    value = tf.gather(colors, indices)

    return value


def colorize_cv(img, vmin = None, vmax = None, cmap = None):
    import numpy as np

    vmin = img.min() if vmin is None else vmin
    vmax = img.max() if vmax is None else vmax
    img = (img - vmin) / (vmax - vmin)
    img = np.squeeze(img)

    indices = np.round(img * 255).astype(np.int32)
    cm = matplotlib.cm.get_cmap(cmap if cmap is not None else 'viridis', lut = 256)
    ret = cm(indices)
    if ret.shape[2] == 4: # image somehow got an extra alpha channel
        ret = ret[:, :, :3] # remove alpha channel since it is not needed and might cause problems
    return np.ascontiguousarray(ret*255, dtype=np.uint8)
