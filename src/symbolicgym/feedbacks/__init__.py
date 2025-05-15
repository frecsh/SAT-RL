"""
Registry for pluggable symbolic feedback backends in SymbolicGym.
"""

FEEDBACK_REGISTRY = {}


def register_feedback(domain_name, feedback_cls):
    FEEDBACK_REGISTRY[domain_name] = feedback_cls


def get_feedback(domain_name):
    return FEEDBACK_REGISTRY.get(domain_name)
