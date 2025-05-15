"""
Cross-domain message interpretation for SymbolicGym.
"""


def interpret_message(message, domain):
    """
    Interpret a message in the context of a domain.
    Args:
        message: any
        domain: str ('sat', 'sympy', 'z3')
    Returns:
        interpretation: str
    """
    if domain == "sat":
        return f"SAT message: {message}"
    elif domain == "sympy":
        return f"SymPy message: {message}"
    elif domain == "z3":
        return f"Z3 message: {message}"
    else:
        return f"Unknown domain: {message}"
