class GestureAction:
    def __init__(self):
        pass

    def call(self):
        raise NotImplementedError("Only Child classes of GestureAction can be used as action callback!")

    def get_name(self):
        raise NotImplementedError("Child classes of GestureAction have to implement get_name()!")
