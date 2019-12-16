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
            conc = np.concatenate((data[1], data[0], pred), axis=2)
            conc = np.squeeze(conc)
            for im in conc:
                fig = plt.figure()

                ax = fig.add_subplot(211)
                ax.set_title("Target --> Augmented Input --> Output")
                ax.imshow(im)

                # ax = fig.add_subplot(212)
                # ax.set_title("Prediction only (scale different!)")
                # localpred = im[:, -int(im.shape[1] / 3):]
                # max_loc = np.unravel_index(np.argmax(localpred, axis=None), localpred.shape)
                # ax.imshow(localpred)
                # ax.annotate('max ({})'.format(localpred[max_loc]),
                #             xy=(max_loc[1], max_loc[0]),
                #             xytext=(int(localpred.shape[1] / 2), int(localpred.shape[0] / 2)),
                #             arrowprops=dict(facecolor='black', shrink=0.05))
                # fig.tight_layout()
                plots.append(plot_to_image(fig))

            with self.writer.as_default():
                tf.summary.image("!A Result plots", plots, step=self.seen, max_outputs=100)


class ConfMapOutputVisualization(tf.keras.callbacks.Callback):

    def __init__(self, log_dir, feed_inputs_display=None, plot_every_x_batches=1000):
        super(ConfMapOutputVisualization, self).__init__()
        self.feed_inputs_display = feed_inputs_display
        self.writer = tf.summary.create_file_writer(log_dir)
        self.plot_every_x_batches = plot_every_x_batches
        self.seen = 0


    def on_batch_end(self, batch, logs=None):
        self.seen += 1
        if self.seen % self.plot_every_x_batches == 0:
            data = self.feed_inputs_display.take(1)
            pred = self.model.predict(data)

            data = data.unbatch()

            min, max = pred.min(), pred.max()
            print("min: {}, max: {}".format(min, max))

            plots = []
            j = 0
            for inp, y_true in data:
                y_pred = pred[j]
                j = j+1
                fig = plt.figure(figsize = (10, 10))
                input = np.squeeze(inp)
                output = np.squeeze(y_pred)
                output = output.transpose([2, 0, 1])
                ax = fig.add_subplot(421)
                ax.set_title("Input")
                ax.imshow(input)
                i = 8
                stacked = input
                for pic in output:
                    ax = fig.add_subplot(4, 7, i)
                    ax.set_title("Joint {}".format(i-7))
                    ax.imshow(pic)
                    stacked = stacked + pic
                    i = i + 1
                ax = fig.add_subplot(422)
                ax.imshow(stacked)
                plots.append(plot_to_image(fig))

            with self.writer.as_default():
                tf.summary.image("!A Result plots", plots, step=self.seen, max_outputs=100)
