import numpy as np
import io

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# make the 1 channel input image or disparity map look good within this color map. This function is not necessary for this Tensorboard problem shown as above. Just a function used in my own research project.
import tools.telegram_tools



# source: https://stackoverflow.com/questions/43784921/how-to-display-custom-images-in-tensorboard-using-keras?rq=1


def colormap_jet(img):
    return cv2.cvtColor(cv2.applyColorMap(np.uint8(img), 2), cv2.COLOR_BGR2RGB)


def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call.
    Source: https://www.tensorflow.org/tensorboard/image_summaries, 27.11.2019
    """
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format = 'png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels = 4)
    # Add the batch dimension
    # image = tf.expand_dims(image, 0)
    return image


class AEOutputVisualization(tf.keras.callbacks.Callback):

    def __init__(self, log_dir, feed_inputs_display = None, plot_every_x_batches = 10):
        super(AEOutputVisualization, self).__init__()
        self.feed_inputs_display = feed_inputs_display
        self.writer = tf.summary.create_file_writer(log_dir)
        self.plot_every_x_batches = plot_every_x_batches
        self.seen = 0


    def on_batch_end(self, batch, logs = None):
        self.seen += 1
        if self.seen % self.plot_every_x_batches == 0:
            data = self.feed_inputs_display.take(1)
            data = next(iter(data))

            pred = self.model.predict(data[0])

            min, max = pred.min(), pred.max()
            print("min: {}, max: {}".format(min, max))

            plots = []
            conc = np.concatenate((data[1], data[0], pred), axis = 2)
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
                tf.summary.image("!A Result plots", plots, step = self.seen, max_outputs = 100)


class ConfMapOutputVisualization(tf.keras.callbacks.Callback):

    def __init__(self, log_dir, feed_inputs_display = None, plot_every_x_batches = 1000, confmap_labels = None,
                 send_telegram = False, telegram_settings = None):
        super(ConfMapOutputVisualization, self).__init__()
        self.feed_inputs_display = feed_inputs_display
        self.writer = tf.summary.create_file_writer(log_dir)
        self.plot_every_x_batches = plot_every_x_batches
        self.seen = 0
        self.confmap_labels = confmap_labels
        self.send_telegram = send_telegram
        self.telegram_settings = telegram_settings

        if self.send_telegram:
            assert "user" in self.telegram_settings.keys()
            assert "token" in self.telegram_settings.keys()
            assert "chat" in self.telegram_settings.keys()

    def on_batch_end(self, batch, logs = None):
        self.seen += 1
        if self.seen % self.plot_every_x_batches == 0:
            data = self.feed_inputs_display.take(1)
            data_eager = next(iter(data))

            pred = self.model.predict(data_eager[0])

            min, max = pred.min(), pred.max()
            # print("min: {}, max: {}".format(min, max))

            plots = []
            sample_cntr = 0
            for inp, y_true, y_pred in zip(data_eager[0], data_eager[1], pred):

                input = np.squeeze(inp)
                output = np.squeeze(y_pred)
                output = output.transpose([2, 0, 1])

                fig = plt.figure(figsize = (12, 12))
                gs = fig.add_gridspec(9, 9)

                i = 0
                stacked = input
                for pic in output:
                    ax = fig.add_subplot(gs[3 + i // 7, 1 + i % 7])
                    if self.confmap_labels is not None:
                        ax.set_title(self.confmap_labels[i])
                    ax.imshow(pic)
                    stacked = stacked + pic
                    i = i + 1

                ax = fig.add_subplot(gs[0:3, 0:3])
                ax.set_title("Input image")
                ax.imshow(input)

                ax = fig.add_subplot(gs[0:3, 3:6])
                ax.set_title("y_pred")
                ax.imshow(stacked)

                ax = fig.add_subplot(gs[0:3, 6:9])
                ax.set_title("y_true")
                stacked = input + np.sum(y_true, axis = 2)
                ax.imshow(stacked)

                fig.tight_layout()

                plots.append(plot_to_image(fig))

                sample_cntr += 1
                if sample_cntr > 20:
                    break

            with self.writer.as_default():
                tf.summary.image("!A Result plots", plots, step = self.seen, max_outputs = 100)

            if self.send_telegram:
                import cv2
                i = 0
                for plot in plots:
                    tools.telegram_tools.telegram_sendPhoto(self.telegram_settings["user"],
                                                            self.telegram_settings["token"],
                                                            self.telegram_settings["chat"],
                                                            cv2.imencode(".png", plot.numpy())[1].tostring(),
                                                            caption = "Batch {}: Illustration Sample: {}".format(
                                                                self.seen, i)
                                                            )
                    i += 1
