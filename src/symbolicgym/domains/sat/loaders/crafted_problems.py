"""Encoder for crafted SAT problems (n-queens, graph-coloring) for transfer testing in SymbolicGym."""


def encode_n_queens(n):
    """Return a SAT problem encoding for the n-queens problem."""
    # Variables: x_{i,j} is True if queen in row i, column j
    # Variable numbers: (i * n + j + 1) for 0-based i, j
    clauses = []
    # One queen per row
    for i in range(n):
        clauses.append([i * n + j + 1 for j in range(n)])
    # One queen per column
    for j in range(n):
        clauses.append([i * n + j + 1 for i in range(n)])
    # At most one queen per row
    for i in range(n):
        for j1 in range(n):
            for j2 in range(j1 + 1, n):
                clauses.append([-(i * n + j1 + 1), -(i * n + j2 + 1)])
    # At most one queen per column
    for j in range(n):
        for i1 in range(n):
            for i2 in range(i1 + 1, n):
                clauses.append([-(i1 * n + j + 1), -(i2 * n + j + 1)])
    # Diagonals
    for i in range(n):
        for j in range(n):
            for d in range(1, n):
                if i + d < n and j + d < n:
                    clauses.append([-(i * n + j + 1), -((i + d) * n + (j + d) + 1)])
                if i + d < n and j - d >= 0:
                    clauses.append([-(i * n + j + 1), -((i + d) * n + (j - d) + 1)])
    num_vars = n * n
    return {"num_vars": num_vars, "clauses": clauses}


def encode_graph_coloring(graph, num_colors):
    """Return a SAT problem encoding for the graph coloring problem.
    graph: dict {node: [neighbors]}
    num_colors: int.
    """
    # Variables: x_{v,c} is True if node v has color c
    # Variable numbers: v * num_colors + c + 1
    nodes = list(graph.keys())
    node_idx = {v: i for i, v in enumerate(nodes)}
    clauses = []
    # Each node has at least one color
    for v in nodes:
        clauses.append([node_idx[v] * num_colors + c + 1 for c in range(num_colors)])
    # Each node has at most one color
    for v in nodes:
        for c1 in range(num_colors):
            for c2 in range(c1 + 1, num_colors):
                clauses.append(
                    [
                        -(node_idx[v] * num_colors + c1 + 1),
                        -(node_idx[v] * num_colors + c2 + 1),
                    ]
                )
    # No two adjacent nodes have the same color
    for v in nodes:
        for u in graph[v]:
            if node_idx[v] < node_idx[u]:  # avoid duplicate constraints
                for c in range(num_colors):
                    clauses.append(
                        [
                            -(node_idx[v] * num_colors + c + 1),
                            -(node_idx[u] * num_colors + c + 1),
                        ]
                    )
    num_vars = len(nodes) * num_colors
    return {"num_vars": num_vars, "clauses": clauses}
