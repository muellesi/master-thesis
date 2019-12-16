import dataclasses
from actions.base_action import GestureAction

@dataclass
class GestureItem:
    name: str
    samples: list
    action: GestureAction