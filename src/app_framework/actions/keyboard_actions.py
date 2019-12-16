from .base_action import GestureAction
from pynput.keyboard import Key, Controller

class KeyboardLeftArrow(GestureAction):

    def __init__(self):
        super.__init__(KeyboardLeftArrow, self)
        self.keyboard = Controller()

    def call(self):
        self.keyboard.press(Key.left)
    
    def get_name(self):
        return "KeyboardLeftArrow"

class KeyboardRightArrow(GestureAction):

    def __init__(self):
        super.__init__(KeyboardRightArrow, self)
        self.keyboard = Controller()

    def call(self):
        self.keyboard.press(Key.left)
    
    def get_name(self):
        return "KeyboardRightArrow"

        
class KeyboardUpArrow(GestureAction):

    def __init__(self):
        super.__init__(KeyboardUpArrow, self)
        self.keyboard = Controller()

    def call(self):
        self.keyboard.press(Key.left)
    
    def get_name(self):
        return "KeyboardUpArrow"

        
class KeyboardDownArrow(GestureAction):

    def __init__(self):
        super.__init__(KeyboardDownArrow, self)
        self.keyboard = Controller()

    def call(self):
        self.keyboard.press(Key.left)
    
    def get_name(self):
        return "KeyboardDownArrow"