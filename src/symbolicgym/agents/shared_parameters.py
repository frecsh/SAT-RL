class SharedParameters:
    """Share parameters across agents for resource optimization."""

    def __init__(self, param_dict):
        self.param_dict = param_dict

    def get(self, key):
        return self.param_dict[key]

    def set(self, key, value):
        self.param_dict[key] = value
