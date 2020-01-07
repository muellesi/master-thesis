import json

import numpy as np

from .actions.base_action import GestureAction
from .gesture_item import GestureItem



def serialize_gesture_collection(collection, filename = 'gestures.json'):
    class ComplexEncoder(json.JSONEncoder):

        def default(self, obj):
            if isinstance(obj, GestureItem):
                return obj.__dict__
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            # Let the base class default method raise the TypeError
            return json.JSONEncoder.default(self, obj)


    serialized = json.dumps(collection, cls = ComplexEncoder)

    with open(filename, 'w') as f:
        f.write(serialized)


def deserialize_to_gesture_collection(filename = 'gestures.json'):
    with open(filename, 'r') as f:
        deserialized = json.load(f)

    new_gestures = []
    for item in deserialized:
        gesture = GestureItem(name = item["name"], samples = [np.array(sample) for sample in item["samples"]],
                              action = item["action"])
        new_gestures.append(gesture)

    return new_gestures
