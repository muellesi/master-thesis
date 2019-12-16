from .base_action import GestureAction
from pynput.keyboard import Key, Controller


class HelloWorldAction(GestureAction):

    def __init__(self):
        super.__init__(HelloWorldAction, self)
        self.keyboard = Controller()

    def call(self):
        with self.keyboard.pressed(Key.cmd):
            self.keyboard.press('r')
        
    
    def get_name(self):
        return "HelloWorldAction"