from .base_action import GestureAction
from pynput.keyboard import Key, Controller as kbd_ctrl

class MediaVolumeMute(GestureAction):

    def __init__(self):
        super.__init__(MediaVolumeMute, self)
        self.keyboard = kbd_ctrl()

    def call(self):
        self.keyboard.press(Key.media_volume_mute)
    
    def get_name(self):
        return "MediaVolumeMute"

class MediaVolumeUp(GestureAction):

    def __init__(self):
        super.__init__(MediaVolumeUp, self)
        self.keyboard = kbd_ctrl()

    def call(self):
        self.keyboard.press(Key.media_volume_up)
    
    def get_name(self):
        return "MediaVolumeUp"

class MediaVolumeDown(GestureAction):

    def __init__(self):
        super.__init__(MediaVolumeDown, self)
        self.keyboard = kbd_ctrl()

    def call(self):
        self.keyboard.press(Key.media_volume_down)
    
    def get_name(self):
        return "MediaVolumeDown"

class MediaPlayPause(GestureAction):

    def __init__(self):
        super.__init__(MediaPlayPause, self)
        self.keyboard = kbd_ctrl()

    def call(self):
        self.keyboard.press(Key.media_play_pause)
    
    def get_name(self):
        return "MediaPlayPause"


class MediaPlayNext(GestureAction):

    def __init__(self):
        super.__init__(MediaPlayNext, self)
        self.keyboard = kbd_ctrl()

    def call(self):
        self.keyboard.press(Key.media_next)
    
    def get_name(self):
        return "MediaPlayNext"


class MediaPlayPrevious(GestureAction):

    def __init__(self):
        super.__init__(MediaPlayPrevious, self)
        self.keyboard = kbd_ctrl()

    def call(self):
        self.keyboard.press(Key.media_previous)
    
    def get_name(self):
        return "MediaPlayPrevious"