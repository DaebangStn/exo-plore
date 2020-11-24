class Dummy:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return self

    def __repr__(self):
        return "<Dummy>"
