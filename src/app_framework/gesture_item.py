import dataclasses
from .actions.base_action import GestureAction



@dataclasses.dataclass
class GestureItem:
    name: str
    samples: list
    action: GestureAction
