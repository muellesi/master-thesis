import platform
from datetime import datetime

import requests
import tensorflow as tf



class PushoverCallback(tf.keras.callbacks.Callback):

    def __init__(self, user, token, training_id = None, training_description = None):
        super().__init__()
        self.predef_message = training_id
        self.user = user,
        self.token = token
        self.hostname = platform.node()
        self.training_description = training_description
        self.training_id = training_id


    def _send_message(self, message):
        url = "https://api.pushover.net/1/messages.json"
        data = { 'user'   : self.user,
                 'token'  : self.token,
                 'sound'  : "intermission",
                 'message': message,
                 'title'  : "Training {} on {}".format("" if self.training_id is None else "#" + str(self.training_id),
                                                       self.hostname)
                 }

        data['message'] = data['message'] + "\n" + str(datetime.now())

        r = requests.post(url = url, data = data)


    def on_epoch_end(self, epoch, logs):
        message = "Epoch: {} ; \n Metrics: {}".format(epoch,
                                                      logs)
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
