import numpy as np


class SATFeedback:
    """
    Feedback vector for SAT environments. Metrics:
    - clause_satisfaction: Fraction of satisfied clauses
    - variable_decisiveness: Fraction of variables assigned
    - search_diversity: Std of variable assignments (exploration)
    - constraint_tension: Avg abs sum of literals per clause (conflict/tightness)
    - proof_progress: clause_satisfaction * variable_decisiveness
    - clause_centrality: Mean degree (variable occurrence) per variable
    - assignment_entropy: Entropy of variable assignment distribution
    - clause_length_var: Variance of clause lengths
    """

    def get_feedback(self, variable_assignment, clauses, num_vars):
        """
        Compute feedback metrics for SAT environments.

        Args:
            variable_assignment (dict): Mapping of variable indices to their assigned values (0, 1, or unassigned).
            clauses (list of list of int): List of clauses, where each clause is a list of literals.
            num_vars (int): Total number of variables in the problem.

        Returns:
            dict: Feedback metrics including clause_satisfaction, variable_decisiveness, search_diversity,
                  constraint_tension, proof_progress, clause_centrality, assignment_entropy, and clause_length_var.
        """
        # Clause satisfaction
        satisfied = [
            any(
                (
                    variable_assignment[abs(lit)] == 1
                    if lit > 0
                    else variable_assignment[abs(lit)] == 0
                )
                for lit in clause
            )
            for clause in clauses
        ]
        clause_satisfaction = sum(satisfied) / len(clauses) if clauses else 0.0

        # Variable decisiveness
        variable_decisiveness = (
            sum([v in (0, 1) for v in variable_assignment.values()]) / num_vars
        )

        # Search diversity
        assignments = np.array(list(variable_assignment.values()))
        search_diversity = float(np.std(assignments))

        # Constraint tension
        constraint_tension = (
            float(np.mean([np.abs(np.sum(clause)) for clause in clauses]))
            if clauses
            else 0.0
        )

        # Proof progress
        proof_progress = clause_satisfaction * variable_decisiveness

        # Clause centrality (mean variable degree)
        var_counts = np.zeros(num_vars)
        for clause in clauses:
            for lit in clause:
                var_counts[abs(lit) - 1] += 1
        clause_centrality = float(np.mean(var_counts)) if num_vars > 0 else 0.0

        # Assignment entropy
        unique, counts = np.unique(assignments, return_counts=True)
        probs = (
            counts / counts.sum()
            if counts.sum() > 0
            else np.ones_like(counts) / len(counts)
        )
        assignment_entropy = float(-np.sum(probs * np.log2(probs + 1e-12)))

        # Clause length variance
        clause_lengths = [len(clause) for clause in clauses]
        clause_length_var = float(np.var(clause_lengths)) if clause_lengths else 0.0

        return {
            "clause_satisfaction": clause_satisfaction,
            "variable_decisiveness": variable_decisiveness,
            "search_diversity": search_diversity,
            "constraint_tension": constraint_tension,
            "proof_progress": proof_progress,
            "clause_centrality": clause_centrality,
            "assignment_entropy": assignment_entropy,
            "clause_length_var": clause_length_var,
        }
