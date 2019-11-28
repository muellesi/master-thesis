import tensorflow as tf
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
import io
import matplotlib.pyplot as plt



# source: https://stackoverflow.com/questions/43784921/how-to-display-custom-images-in-tensorboard-using-keras?rq=1

# make the 1 channel input image or disparity map look good within this color map. This function is not necessary for this Tensorboard problem shown as above. Just a function used in my own research project.
def colormap_jet(img):
    return cv2.cvtColor(cv2.applyColorMap(np.uint8(img), 2), cv2.COLOR_BGR2RGB)


def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call.
    Source: https://www.tensorflow.org/tensorboard/image_summaries, 27.11.2019
    """
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    # image = tf.expand_dims(image, 0)
    return image


class AEOutputVisualization(tf.keras.callbacks.Callback):

    def __init__(self, log_dir, feed_inputs_display=None, plot_every_x_batches=10):
        super(AEOutputVisualization, self).__init__()
        self.feed_inputs_display = feed_inputs_display
        self.writer = tf.summary.create_file_writer(log_dir)
        self.plot_every_x_batches = plot_every_x_batches
        self.seen = 0

    def on_batch_end(self, batch, logs=None):
        self.seen += 1
        if self.seen % self.plot_every_x_batches == 0:
            data = self.feed_inputs_display.take(1)
            data = next(iter(data))

            pred = self.model.predict(data[0])

            min, max = pred.min(), pred.max()
            print("min: {}, max: {}".format(min, max))

            plots = []
            conc = np.concatenate((data[0], pred), axis=2)
            conc = np.squeeze(conc)
            for im in conc:
                fig = plt.figure()
                ax = fig.add_subplot(111)
                ax.set_title("Original --> Output")
                ax.imshow(im)
                plots.append(plot_to_image(fig))

            with self.writer.as_default():
                tf.summary.image("Result plots", plots, step=self.seen, max_outputs=100)
