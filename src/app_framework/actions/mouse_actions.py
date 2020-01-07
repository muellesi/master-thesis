from pynput.mouse import Button
from pynput.mouse import Controller

from .base_action import GestureAction



class MouseLeftClick(GestureAction):

    def __init__(self):
        super().__init__()
        self.mouse = Controller()


    def call(self):
        self.mouse.click(Button.left, 1)


    def get_name(self):
        return "MouseLeftClick"


class MouseRightClick(GestureAction):

    def __init__(self):
        super().__init__()
        self.mouse = Controller()


    def call(self):
        # perform mouse right click
        self.mouse.click(Button.right, 1)


    def get_name(self):
        return "MouseRightClick"


class MouseDoubleClick(GestureAction):

    def __init__(self):
        super().__init__()
        self.mouse = Controller()


    def call(self):
        self.mouse.click(Button.left, 2)


    def get_name(self):
        return "MouseDoubleClick"
