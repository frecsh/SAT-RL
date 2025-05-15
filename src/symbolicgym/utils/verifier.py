from sympy import solve, sympify


def is_valid_problem(expr_str):
    try:
        expr = sympify(expr_str)
        sol = solve(expr)
        return bool(sol)
    except Exception:
        return False
