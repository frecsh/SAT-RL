"""
Equation transformation graph visualization for SymPy domain.
"""
import matplotlib.pyplot as plt
import networkx as nx


def plot_transformation_graph(transformations, ax=None):
    """
    Plot a graph of equation transformations.
    Args:
        transformations: list of (from_expr, to_expr, label)
        ax: matplotlib axis (optional)
    """
    G = nx.DiGraph()
    for from_expr, to_expr, label in transformations:
        G.add_edge(str(from_expr), str(to_expr), label=label)
    pos = nx.spring_layout(G)
    if ax is None:
        fig, ax = plt.subplots()
    nx.draw(G, pos, ax=ax, with_labels=True, node_color="lightblue")
    edge_labels = nx.get_edge_attributes(G, "label")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax)
    ax.set_title("Equation Transformation Graph")
    return ax
