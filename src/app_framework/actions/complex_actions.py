import pyttsx3
from pynput.keyboard import Controller

from .base_action import GestureAction



class HelloWorldAction(GestureAction):

    def __init__(self):
        super().__init__()
        self.keyboard = Controller()


    def call(self):
        # with self.keyboard.pressed(Key.cmd):
        #    self.keyboard.press('r')
        engine = pyttsx3.init()
        engine.say('Hello World!')
        engine.runAndWait()


    def get_name(self):
        return "HelloWorldAction"
