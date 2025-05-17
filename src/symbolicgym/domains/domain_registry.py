"""Domain registration API for extensibility."""


class DomainRegistry:
    _registry = {}

    @classmethod
    def register(cls, name, domain_cls):
        cls._registry[name] = domain_cls

    @classmethod
    def get(cls, name):
        return cls._registry[name]

    @classmethod
    def list_domains(cls):
        return list(cls._registry.keys())
