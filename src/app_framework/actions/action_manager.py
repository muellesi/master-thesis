from .complex_actions import HelloWorldAction
from .keyboard_actions import KeyboardDownArrow
from .keyboard_actions import KeyboardLeftArrow
from .keyboard_actions import KeyboardRightArrow
from .keyboard_actions import KeyboardUpArrow
from .media_actions import MediaPlayNext
from .media_actions import MediaPlayPause
from .media_actions import MediaPlayPrevious
from .media_actions import MediaVolumeDown
from .media_actions import MediaVolumeMute
from .media_actions import MediaVolumeUp
from .mouse_actions import MouseDoubleClick
from .mouse_actions import MouseLeftClick
from .mouse_actions import MouseRightClick



class ActionManager:

    def __init__(self):
        _actions = [
                MouseLeftClick(),
                MouseRightClick(),
                MouseDoubleClick(),
                KeyboardLeftArrow(),
                KeyboardRightArrow(),
                KeyboardUpArrow(),
                KeyboardDownArrow(),
                MediaPlayNext(),
                MediaPlayPause(),
                MediaPlayPrevious(),
                MediaVolumeDown(),
                MediaVolumeMute(),
                MediaVolumeUp(),
                HelloWorldAction()
                ]

        self._actions = { }
        for action in _actions:
            if action.get_name() in self._actions:
                raise ValueError("Action {} already exists!".format(action.get_name()))
            self._actions[action.get_name()] = action


    def get_action_names(self):
        return [action for action in self._actions]

    def get_action_by_name(self, action_name):
        return self._actions.get(action_name)

    def exec_action(self, action_name):
        act = self._actions.get(action_name)
        if act:
            act.call()
