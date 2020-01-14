import win32com.client as wincl
from pynput.keyboard import Controller

from .base_action import GestureAction



class TTSAction(GestureAction):

    def __init__(self, text = "Hello World"):
        super().__init__()
        self.keyboard = Controller()
        self.text = text


    def call(self):
        # with self.keyboard.pressed(Key.cmd):
        #    self.keyboard.press('r')
        speak = wincl.Dispatch("SAPI.SpVoice")
        speak.Speak(self.text)


    def get_name(self):
        return "TTS<\"{}\">".format(self.text)
