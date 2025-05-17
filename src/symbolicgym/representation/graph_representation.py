"""Incidence-graph encoding for SAT problems (variables <-> clauses)."""


def incidence_graph(clauses, num_vars):
    """Build a bipartite incidence graph (variables <-> clauses) for a SAT problem.

    Args:
        clauses: List of lists, each inner list is a clause (ints, positive/negative for polarity)
        num_vars: Number of variables (int)

    Returns:
        nodes: List of node labels (str: 'v1', 'v2', ..., 'c1', ...)
        edges: List of (node_idx, node_idx) tuples
    """
    nodes = [f"v{i + 1}" for i in range(num_vars)] + [
        f"c{j + 1}" for j in range(len(clauses))
    ]
    edges = []
    for j, clause in enumerate(clauses):
        for lit in clause:
            var_idx = abs(lit) - 1
            clause_idx = num_vars + j
            edges.append((var_idx, clause_idx))
    return nodes, edges
