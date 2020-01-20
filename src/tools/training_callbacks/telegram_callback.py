import platform
from datetime import datetime

import requests
import tensorflow as tf
import pprint


def _telegram_escape(message):
    must_escape = ['_', '*', '[', ']', '(', ')', '~', '`', '>', '#', '+', '-', '=', '|', '{', '}', '.', '!']
    for char in must_escape:
        message = message.replace(char, '\\' + char)
    return message


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
        url = "https://api.telegram.org/bot{}:{}/sendMessage".format(self.user, self.token)

        message_title = "Training {} on #{}".format("" if self.training_id is None else "#" + str(self.training_id),
                                                    self.hostname)

        composed_message = "*{title}*\n\n{message_body}\n\n_{timestamp}_".format(
                title = _telegram_escape(message_title),
                message_body = _telegram_escape(message),
                timestamp = _telegram_escape(str(datetime.now()))
                )

        data = { 'chat_id'   : self.chat_id,
                 'text'      : composed_message,
                 'parse_mode': "MarkdownV2",
                 }

        r = requests.post(url = url, json = data)


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


    def on_train_end(self, logs = None):
        if self.training_id is not None:
            message = "Training {} ends!".format(self.training_id)
        else:
            message = "Training ends!"

        self._send_message(message)
