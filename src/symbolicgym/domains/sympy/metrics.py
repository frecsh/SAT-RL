"""SymPy domain metrics for SymbolicGym."""


class SymPyMetrics:
    """Stateful SymPy metric collector for episodic RL and interpretability."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.original_expr_size = 0
        self.final_expr_size = 0
        self.proof_steps = 0
        self.optimal_steps = 0

    def set_original_expr(self, expr):
        self.original_expr_size = len(str(expr))

    def set_final_expr(self, expr):
        self.final_expr_size = len(str(expr))

    def add_proof_step(self):
        self.proof_steps += 1

    def set_optimal_steps(self, n):
        self.optimal_steps = n

    def compute(self):
        simplification_ratio = (
            (self.original_expr_size - self.final_expr_size) / self.original_expr_size
            if self.original_expr_size
            else 0
        )
        proof_step_efficiency = self.optimal_steps / max(1, self.proof_steps)
        return {
            "simplification_ratio": simplification_ratio,
            "proof_step_efficiency": proof_step_efficiency,
        }


def sympy_simplification_ratio(original_expr, simplified_expr):
    return float(len(str(original_expr))) / max(1, len(str(simplified_expr)))


def sympy_proof_step_efficiency(num_steps, optimal_steps):
    return optimal_steps / max(1, num_steps)
