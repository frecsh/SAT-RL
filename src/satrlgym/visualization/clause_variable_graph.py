"""
Clause-variable graph visualization.

This module provides a visualization of CNF formulas as bipartite graphs,
where nodes are variables and clauses and edges represent the inclusion of
variables in clauses.
"""


import matplotlib.pyplot as plt
import networkx as nx


class ClauseVariableGraph:
    """
    Bipartite graph visualization of a CNF formula.

    Attributes:
        num_variables: Number of variables in the formula
        num_clauses: Number of clauses in the formula
        graph: NetworkX graph object
        positions: Node positions for visualization
    """

    def __init__(self, clauses: list[list[int]], num_vars: int):
        """
        Initialize the graph from a CNF formula.

        Args:
            clauses: List of clauses, each clause being a list of literals
            num_vars: Number of variables in the formula
        """
        self.num_variables = num_vars
        self.num_clauses = len(clauses)

        # Create a bipartite graph
        self.graph = nx.Graph()

        # Add variable nodes
        for var in range(1, num_vars + 1):
            self.graph.add_node(f"v{var}", type="variable", label=str(var))

        # Add clause nodes
        for i, clause in enumerate(clauses):
            clause_id = f"c{i+1}"
            self.graph.add_node(clause_id, type="clause", label=f"C{i+1}")

            # Add edges to variables in this clause
            for lit in clause:
                var = abs(lit)
                if var <= num_vars:
                    var_id = f"v{var}"
                    # Edge weight indicates positive/negative literal
                    weight = 1 if lit > 0 else -1
                    self.graph.add_edge(clause_id, var_id, weight=weight)

        # Compute layout
        self._compute_layout()

    def _compute_layout(self):
        """
        Compute a layout for the graph visualization.
        """
        # Create sets of variable and clause nodes
        var_nodes = [
            node
            for node, attrs in self.graph.nodes(data=True)
            if attrs.get("type") == "variable"
        ]
        clause_nodes = [
            node
            for node, attrs in self.graph.nodes(data=True)
            if attrs.get("type") == "clause"
        ]

        # Use bipartite layout for better visualization
        self.positions = nx.bipartite_layout(
            self.graph, nodes=var_nodes, align="vertical", scale=2.0
        )

    def update_with_assignment(
        self, assignment: dict[int, bool], satisfied_clauses: list[int] | None = None
    ):
        """
        Update the graph with a variable assignment.

        Args:
            assignment: Dictionary mapping variable indices to Boolean values
            satisfied_clauses: Optional list of indices of satisfied clauses
        """
        # Reset node colors
        nx.set_node_attributes(self.graph, "lightblue", name="color")

        # Update variable nodes
        for var, value in assignment.items():
            var_id = f"v{var}"
            if var_id in self.graph:
                color = "green" if value else "red"
                self.graph.nodes[var_id]["color"] = color

        # Update clause nodes if provided
        if satisfied_clauses is not None:
            for i in satisfied_clauses:
                clause_id = f"c{i+1}"
                if clause_id in self.graph:
                    self.graph.nodes[clause_id]["color"] = "green"

    def draw(self, ax=None, **kwargs):
        """
        Draw the graph.

        Args:
            ax: Matplotlib axes to draw on
            **kwargs: Additional kwargs to pass to nx.draw

        Returns:
            The matplotlib axes with the drawn graph
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))

        # Get node colors
        node_colors = [
            data.get("color", "lightblue") for _, data in self.graph.nodes(data=True)
        ]

        # Get edge colors based on positive/negative literals
        edge_colors = [
            "blue" if data.get("weight", 1) > 0 else "red"
            for _, _, data in self.graph.edges(data=True)
        ]

        # Draw nodes
        nx.draw_networkx_nodes(
            self.graph, self.positions, node_color=node_colors, ax=ax, **kwargs
        )

        # Draw edges
        nx.draw_networkx_edges(
            self.graph, self.positions, edge_color=edge_colors, ax=ax, **kwargs
        )

        # Draw labels
        nx.draw_networkx_labels(
            self.graph,
            self.positions,
            labels={
                node: data.get("label", "")
                for node, data in self.graph.nodes(data=True)
            },
            ax=ax,
            **kwargs,
        )

        ax.set_title("Clause-Variable Graph")
        ax.axis("off")

        return ax
