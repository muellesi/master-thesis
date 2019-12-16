from .mouse_actions import MouseLeftClick, MouseRightClick, MouseDoubleClick
from .keyboard_actions import KeyboardLeftArrow, KeyboardRightArrow, KeyboardUpArrow, KeyboardDownArrow
from .media_actions import MediaPlayNext, MediaPlayPause, MediaPlayPrevious, MediaVolumeDown, MediaVolumeMute, MediaVolumeUp

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
            MediaVolumeUp()
        ]

        self._actions = {}
        for action in _actions:
            if action.get_name() in self._actions:
                raise ValueError("Action {} already exists!".format(action.get_name()))
            self._actions[action.get_name()] = action
    

    def get_action_names(self):
        return [action for action in self._actions]
    

    def exec_action(self, action_name):
        act = self._actions.get(action_name)
        if act:
            act.call()