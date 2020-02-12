import platform
import pprint

import tensorflow as tf

import tools
import tools.telegram_tools



logger = tools.get_logger(__name__, do_file_logging = True)


class TelegramCallback(tf.keras.callbacks.Callback):

    def __init__(self, user, token, chat_id, training_id = None, training_description = None):
        super().__init__()
        self.user = user
        self.token = token
        self.hostname = platform.node()
        self.training_description = training_description
        self.training_id = training_id
        self.chat_id = chat_id

    def _send_message(self, message):
        message_title = "Training {} on #{}".format("" if self.training_id is None else "#" + str(self.training_id),
                                                    self.hostname)

        tools.telegram_tools.telegram_sendMessage(self.user, self.token, self.chat_id, message, message_title)


    def on_epoch_end(self, epoch, logs):
        message = "Epoch: {} ; \n Metrics: {}".format(epoch,
                                                      pprint.pformat(logs))

        self._send_message(message)


    def on_train_begin(self, logs = None):
        if self.training_id is not None:
            message = "Training {} begins!".format(self.training_id)
        else:
            message = "Training begins!"

        if self.training_description is not None:
            message += "\n"
            message += str(self.training_description)
        self._send_message(message)


    def on_test_begin(self, logs = None):
        if self.training_id is not None:
            message = "{}: Start validation!".format(self.training_id)
        else:
            message = "Start validation!"
        self._send_message(message)


    def on_test_end(self, logs = None):
        if self.training_id is not None:
            message = "{}: End validation!".format(self.training_id)
        else:
            message = "End validation!"
        self._send_message(message)


    def on_train_end(self, logs = None):
        if self.training_id is not None:
            message = "Training {} ends!".format(self.training_id)
        else:
            message = "Training ends!"

        self._send_message(message)
