"""NetworkX graph output for SAT problems (variable and clause nodes)."""

import networkx as nx


def sat_to_networkx(num_vars, clauses):
    """Return a NetworkX graph with variable and clause nodes, edges labeled by sign."""
    G = nx.Graph()
    for v in range(1, num_vars + 1):
        G.add_node(f"v{v}", type="variable")
    for i, clause in enumerate(clauses):
        c_name = f"c{i}"
        G.add_node(c_name, type="clause")
        for lit in clause:
            v = abs(lit)
            G.add_edge(f"v{v}", c_name, sign=(lit > 0))
    return G


def sat_incidence_networkx(clauses, num_vars):
    """Return a NetworkX bipartite graph for SAT incidence structure.

    Args:
        clauses: List of lists, each inner list is a clause (ints, positive/negative for polarity)
        num_vars: Number of variables (int)

    Returns:
        G: networkx.Graph (bipartite)
    """
    G = nx.Graph()
    for i in range(num_vars):
        G.add_node(f"v{i + 1}", bipartite=0)
    for j, clause in enumerate(clauses):
        c_name = f"c{j + 1}"
        G.add_node(c_name, bipartite=1)
        for lit in clause:
            G.add_edge(f"v{abs(lit)}", c_name, polarity=int(lit > 0))
    return G
