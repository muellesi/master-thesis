from pynput.keyboard import Controller
from pynput.keyboard import Key

from .base_action import GestureAction



class KeyboardLeftArrow(GestureAction):

    def __init__(self):
        super().__init__()
        self.keyboard = Controller()


    def call(self):
        self.keyboard.press(Key.left)


    def get_name(self):
        return "KeyboardLeftArrow"


class KeyboardRightArrow(GestureAction):

    def __init__(self):
        super().__init__()
        self.keyboard = Controller()


    def call(self):
        self.keyboard.press(Key.right)


    def get_name(self):
        return "KeyboardRightArrow"


class KeyboardUpArrow(GestureAction):

    def __init__(self):
        super().__init__()
        self.keyboard = Controller()


    def call(self):
        self.keyboard.press(Key.left)


    def get_name(self):
        return "KeyboardUpArrow"


class KeyboardDownArrow(GestureAction):

    def __init__(self):
        super().__init__()
        self.keyboard = Controller()


    def call(self):
        self.keyboard.press(Key.left)


    def get_name(self):
        return "KeyboardDownArrow"
