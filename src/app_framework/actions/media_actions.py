from pynput.keyboard import Controller as kbd_ctrl
from pynput.keyboard import Key

from .base_action import GestureAction



class MediaVolumeMute(GestureAction):

    def __init__(self):
        super().__init__()
        self.keyboard = kbd_ctrl()


    def call(self):
        self.keyboard.press(Key.media_volume_mute)


    def get_name(self):
        return "MediaVolumeMute"


class MediaVolumeUp(GestureAction):

    def __init__(self):
        super().__init__()
        self.keyboard = kbd_ctrl()


    def call(self):
        self.keyboard.press(Key.media_volume_up)


    def get_name(self):
        return "MediaVolumeUp"


class MediaVolumeDown(GestureAction):

    def __init__(self):
        super().__init__()
        self.keyboard = kbd_ctrl()


    def call(self):
        self.keyboard.press(Key.media_volume_down)


    def get_name(self):
        return "MediaVolumeDown"


class MediaPlayPause(GestureAction):

    def __init__(self):
        super().__init__()
        self.keyboard = kbd_ctrl()


    def call(self):
        self.keyboard.press(Key.media_play_pause)


    def get_name(self):
        return "MediaPlayPause"


class MediaPlayNext(GestureAction):

    def __init__(self):
        super().__init__()
        self.keyboard = kbd_ctrl()


    def call(self):
        self.keyboard.press(Key.media_next)


    def get_name(self):
        return "MediaPlayNext"


class MediaPlayPrevious(GestureAction):

    def __init__(self):
        super().__init__()
        self.keyboard = kbd_ctrl()


    def call(self):
        self.keyboard.press(Key.media_previous)


    def get_name(self):
        return "MediaPlayPrevious"
