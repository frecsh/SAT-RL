from sympy import sympify


def get_symbolic_feedback(expression):
    expr = sympify(expression)
    # Example features
    is_solved = int(expr.is_Equality and str(expr.lhs) == "x" and expr.rhs.is_number)
    degree = expr.as_poly().degree() if expr.is_polynomial() else 0
    num_terms = len(expr.as_ordered_terms()) if hasattr(expr, "as_ordered_terms") else 1
    variable_on_lhs = int(expr.is_Equality and str(expr.lhs) == "x")
    return {
        "is_solved": is_solved,
        "degree": degree,
        "num_terms": num_terms,
        "variable_on_lhs": variable_on_lhs,
    }
