"""
Graph Neural Network Representation for SAT Problems.

This module implements a GNN-based representation of SAT problems as bipartite graphs,
with variables and clauses as nodes and edges representing variable membership in clauses.
"""


import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphBuilder:
    """Constructs bipartite graphs from SAT problems."""

    def __init__(
        self, num_variables: int, num_clauses: int, with_edge_features: bool = True
    ):
        """
        Initialize the graph builder.

        Args:
            num_variables: Number of variables in the SAT problem
            num_clauses: Number of clauses in the SAT problem
            with_edge_features: Whether to include edge features (polarity information)
        """
        self.num_variables = num_variables
        self.num_clauses = num_clauses
        self.with_edge_features = with_edge_features

    def build_bipartite_graph(
        self,
        clauses: list[list[int]],
        var_assignments: dict[int, bool] | None = None,
    ) -> dict:
        """
        Build a bipartite graph representation of a SAT problem.

        Args:
            clauses: List of clauses, where each clause is a list of literals
                     (integers, positive or negative for regular or negated variables)
            var_assignments: Optional dictionary of variable assignments

        Returns:
            Dictionary containing the graph structure as PyTorch tensors:
                - variable_features: Features for variable nodes [num_variables, var_feature_dim]
                - clause_features: Features for clause nodes [num_clauses, clause_feature_dim]
                - edges: Tensor of shape [2, num_edges] representing (var_idx, clause_idx) pairs
                - edge_features: Edge features [num_edges, edge_feature_dim]
        """
        # Initialize features
        # Variable features: [unassigned, assigned false, assigned true]
        variable_features = torch.zeros(self.num_variables, 3, dtype=torch.float32)
        variable_features[:, 0] = 1.0  # Default to unassigned

        if var_assignments:
            for var_idx, is_true in var_assignments.items():
                # Convert to 0-indexed for internal representation
                var_idx_0 = var_idx - 1 if var_idx > 0 else abs(var_idx) - 1
                # Set the appropriate feature
                variable_features[var_idx_0, 0] = 0.0  # Not unassigned
                if is_true:
                    variable_features[var_idx_0, 2] = 1.0  # Assigned true
                else:
                    variable_features[var_idx_0, 1] = 1.0  # Assigned false

        # Clause features: [unsatisfied, satisfied, unknown]
        clause_features = torch.zeros(self.num_clauses, 3, dtype=torch.float32)
        clause_features[:, 2] = 1.0  # Default to unknown satisfaction

        # Edges: (var_idx, clause_idx) pairs and edge features
        var_clause_edges = []
        edge_features_list = []

        for clause_idx, clause in enumerate(clauses):
            clause_satisfied = False
            clause_unknown = False

            for literal in clause:
                var_idx = abs(literal) - 1  # Convert to 0-indexed
                is_negated = literal < 0

                # Add edge between variable and clause
                var_clause_edges.append((var_idx, clause_idx))

                # Edge feature: is the literal negated or not?
                if self.with_edge_features:
                    edge_features_list.append([1.0, 0.0] if is_negated else [0.0, 1.0])

                # Check if the clause is satisfied by this literal
                if var_assignments and abs(literal) in var_assignments:
                    assigned_value = var_assignments[abs(literal)]
                    if (not is_negated and assigned_value) or (
                        is_negated and not assigned_value
                    ):
                        clause_satisfied = True
                else:
                    clause_unknown = True

            # Update clause satisfaction status
            if clause_satisfied:
                clause_features[clause_idx] = torch.tensor([0.0, 1.0, 0.0])
            elif not clause_unknown:  # All variables assigned but clause not satisfied
                clause_features[clause_idx] = torch.tensor([1.0, 0.0, 0.0])

        # Convert edges list to tensor
        if var_clause_edges:
            edges = torch.tensor(
                var_clause_edges, dtype=torch.long
            ).t()  # Shape [2, num_edges]
            if self.with_edge_features:
                edge_features = torch.tensor(edge_features_list, dtype=torch.float32)
            else:
                edge_features = None
        else:
            # Handle empty graph case
            edges = torch.zeros(2, 0, dtype=torch.long)
            edge_features = (
                torch.zeros(0, 2, dtype=torch.float32)
                if self.with_edge_features
                else None
            )

        return {
            "variable_features": variable_features,
            "clause_features": clause_features,
            "edges": edges,
            "edge_features": edge_features,
        }

    def build_networkx_graph(
        self,
        clauses: list[list[int]],
        var_assignments: dict[int, bool] | None = None,
    ) -> nx.Graph:
        """
        Build a NetworkX graph for visualization and analysis.

        Args:
            clauses: List of clauses, where each clause is a list of literals
            var_assignments: Optional dictionary of variable assignments

        Returns:
            NetworkX graph with variable and clause nodes
        """
        G = nx.Graph()

        # Add variable nodes
        for var_idx in range(1, self.num_variables + 1):
            assignment = None
            if var_assignments and var_idx in var_assignments:
                assignment = var_assignments[var_idx]

            G.add_node(
                f"v{var_idx}",
                bipartite=0,
                type="variable",
                assigned=assignment is not None,
                value=assignment,
            )

        # Add clause nodes and edges
        for clause_idx, clause in enumerate(clauses):
            G.add_node(f"c{clause_idx}", bipartite=1, type="clause")

            satisfied = False
            if var_assignments:
                # Check if clause is satisfied
                for lit in clause:
                    var_idx = abs(lit)
                    if var_idx in var_assignments:
                        is_negated = lit < 0
                        assigned_value = var_assignments[var_idx]
                        if (not is_negated and assigned_value) or (
                            is_negated and not assigned_value
                        ):
                            satisfied = True
                            break

            G.nodes[f"c{clause_idx}"]["satisfied"] = satisfied

            for lit in clause:
                var_idx = abs(lit)
                is_negated = lit < 0
                G.add_edge(f"v{var_idx}", f"c{clause_idx}", negated=is_negated)

        return G


class MessagePassingLayer(nn.Module):
    """
    Message passing layer for variable-clause graph.
    Implements distinct update functions for variable->clause and clause->variable messages.
    """

    def __init__(
        self,
        var_feature_dim: int,
        clause_feature_dim: int,
        message_dim: int = 64,
        edge_feature_dim: int = 2,
    ):
        """
        Initialize the message passing layer.

        Args:
            var_feature_dim: Dimensionality of variable node features
            clause_feature_dim: Dimensionality of clause node features
            message_dim: Dimensionality of messages passed between nodes
            edge_feature_dim: Dimensionality of edge features
        """
        super().__init__()

        # Networks for variable -> clause messages
        self.var_to_clause_network = nn.Sequential(
            nn.Linear(var_feature_dim + edge_feature_dim, message_dim),
            nn.ReLU(),
            nn.Linear(message_dim, message_dim),
        )

        # Networks for clause -> variable messages
        self.clause_to_var_network = nn.Sequential(
            nn.Linear(clause_feature_dim + edge_feature_dim, message_dim),
            nn.ReLU(),
            nn.Linear(message_dim, message_dim),
        )

        # Update networks
        self.var_update_network = nn.GRUCell(message_dim, var_feature_dim)
        self.clause_update_network = nn.GRUCell(message_dim, clause_feature_dim)

        # Attention for focusing on unsatisfied clauses
        self.attention_network = nn.Sequential(
            nn.Linear(clause_feature_dim, message_dim),
            nn.Tanh(),
            nn.Linear(message_dim, 1),
        )

    def forward(
        self,
        var_features: torch.Tensor,
        clause_features: torch.Tensor,
        edges: torch.Tensor,
        edge_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the message passing layer.

        Args:
            var_features: Variable node features [num_variables, var_feature_dim]
            clause_features: Clause node features [num_clauses, clause_feature_dim]
            edges: Edge indices [2, num_edges] as (var_idx, clause_idx) pairs
            edge_features: Optional edge features [num_edges, edge_feature_dim]

        Returns:
            Updated variable and clause features
        """
        if edges.shape[1] == 0:  # No edges case
            return var_features, clause_features

        num_variables = var_features.shape[0]
        num_clauses = clause_features.shape[0]
        device = var_features.device

        # Default edge features if not provided
        if edge_features is None:
            edge_features = torch.ones(edges.shape[1], 2, device=device) / 2

        # Step 1: Compute variable -> clause messages
        var_indices = edges[0]  # [num_edges]
        clause_indices = edges[1]  # [num_edges]

        var_to_clause_inputs = torch.cat(
            [var_features[var_indices], edge_features], dim=1
        )  # [num_edges, var_feature_dim + edge_feature_dim]
        var_to_clause_messages = self.var_to_clause_network(
            var_to_clause_inputs
        )  # [num_edges, message_dim]

        # Step 2: Compute attention weights for clauses (focusing on unsatisfied clauses)
        clause_attention = self.attention_network(clause_features).squeeze(
            -1
        )  # [num_clauses]
        clause_attention = F.softmax(clause_attention, dim=0)  # [num_clauses]

        # Apply attention to messages going to clauses
        var_to_clause_messages = var_to_clause_messages * clause_attention[
            clause_indices
        ].unsqueeze(
            1
        )  # [num_edges, message_dim]

        # Step 3: Aggregate messages for each clause
        clause_messages = torch.zeros(
            num_clauses, var_to_clause_messages.shape[1], device=device
        )
        clause_messages.index_add_(0, clause_indices, var_to_clause_messages)

        # Step 4: Update clause features
        clause_features_new = self.clause_update_network(
            clause_messages, clause_features
        )  # [num_clauses, clause_feature_dim]

        # Step 5: Compute clause -> variable messages
        clause_to_var_inputs = torch.cat(
            [clause_features[clause_indices], edge_features], dim=1
        )  # [num_edges, clause_feature_dim + edge_feature_dim]
        clause_to_var_messages = self.clause_to_var_network(
            clause_to_var_inputs
        )  # [num_edges, message_dim]

        # Step 6: Aggregate messages for each variable
        var_messages = torch.zeros(
            num_variables, clause_to_var_messages.shape[1], device=device
        )
        var_messages.index_add_(0, var_indices, clause_to_var_messages)

        # Step 7: Update variable features
        var_features_new = self.var_update_network(
            var_messages, var_features
        )  # [num_variables, var_feature_dim]

        return var_features_new, clause_features_new


class SATGraphNN(nn.Module):
    """
    Graph Neural Network for SAT problem representation.
    Creates a learnable embedding of the SAT problem structure.
    """

    def __init__(
        self,
        num_variables: int,
        num_clauses: int,
        var_feature_dim: int = 3,
        clause_feature_dim: int = 3,
        message_dim: int = 64,
        edge_feature_dim: int = 2,
        num_message_passing_steps: int = 3,
        output_dim: int = 128,
        graph_pooling: str = "attention",
    ):
        """
        Initialize the SAT Graph Neural Network.

        Args:
            num_variables: Number of variables in the SAT problem
            num_clauses: Number of clauses in the SAT problem
            var_feature_dim: Dimensionality of variable node features
            clause_feature_dim: Dimensionality of clause node features
            message_dim: Dimensionality of messages passed between nodes
            edge_feature_dim: Dimensionality of edge features
            num_message_passing_steps: Number of message passing iterations
            output_dim: Dimensionality of the final graph representation
            graph_pooling: Method for global pooling ('mean', 'sum', 'attention')
        """
        super().__init__()

        self.num_variables = num_variables
        self.num_clauses = num_clauses
        self.var_feature_dim = var_feature_dim
        self.clause_feature_dim = clause_feature_dim
        self.num_message_passing_steps = num_message_passing_steps
        self.graph_pooling = graph_pooling

        # Graph builder for creating bipartite graphs
        self.graph_builder = GraphBuilder(
            num_variables, num_clauses, with_edge_features=True
        )

        # Message passing layers
        self.message_passing_layers = nn.ModuleList(
            [
                MessagePassingLayer(
                    var_feature_dim, clause_feature_dim, message_dim, edge_feature_dim
                )
                for _ in range(num_message_passing_steps)
            ]
        )

        # Graph-level readout - variable node projection
        self.var_projection = nn.Linear(var_feature_dim, output_dim)

        # Graph-level readout - clause node projection
        self.clause_projection = nn.Linear(clause_feature_dim, output_dim)

        # Attention for graph pooling (if needed)
        if graph_pooling == "attention":
            self.var_attention = nn.Sequential(
                nn.Linear(var_feature_dim, message_dim),
                nn.Tanh(),
                nn.Linear(message_dim, 1),
            )

            self.clause_attention = nn.Sequential(
                nn.Linear(clause_feature_dim, message_dim),
                nn.Tanh(),
                nn.Linear(message_dim, 1),
            )

    def forward(self, sat_problem: dict) -> dict[str, torch.Tensor]:
        """
        Process a SAT problem through the graph neural network.

        Args:
            sat_problem: Dictionary with keys:
                - 'clauses': List of clauses, each a list of literals
                - 'variable_assignments': Dictionary of current variable assignments

        Returns:
            Dictionary with different graph representations:
                - 'graph_embedding': Global graph representation
                - 'variable_embeddings': Per-variable node embeddings
                - 'clause_embeddings': Per-clause node embeddings
        """
        # Extract problem components
        clauses = sat_problem["clauses"]
        var_assignments = sat_problem.get("variable_assignments", {})

        # Build the graph
        graph_data = self.graph_builder.build_bipartite_graph(clauses, var_assignments)

        var_features = graph_data["variable_features"]
        clause_features = graph_data["clause_features"]
        edges = graph_data["edges"]
        edge_features = graph_data["edge_features"]

        # Move to the same device as the model
        device = next(self.parameters()).device
        var_features = var_features.to(device)
        clause_features = clause_features.to(device)
        edges = edges.to(device)
        if edge_features is not None:
            edge_features = edge_features.to(device)

        # Apply message passing
        for layer in self.message_passing_layers:
            var_features, clause_features = layer(
                var_features, clause_features, edges, edge_features
            )

        # Project node features
        var_embeddings = self.var_projection(
            var_features
        )  # [num_variables, output_dim]
        clause_embeddings = self.clause_projection(
            clause_features
        )  # [num_clauses, output_dim]

        # Perform graph-level pooling
        if self.graph_pooling == "mean":
            var_pooled = torch.mean(var_embeddings, dim=0)  # [output_dim]
            clause_pooled = torch.mean(clause_embeddings, dim=0)  # [output_dim]
            graph_embedding = (var_pooled + clause_pooled) / 2  # [output_dim]

        elif self.graph_pooling == "sum":
            var_pooled = torch.sum(var_embeddings, dim=0)  # [output_dim]
            clause_pooled = torch.sum(clause_embeddings, dim=0)  # [output_dim]
            graph_embedding = var_pooled + clause_pooled  # [output_dim]

        elif self.graph_pooling == "attention":
            # Compute attention weights
            var_weights = self.var_attention(var_features).squeeze(
                -1
            )  # [num_variables]
            clause_weights = self.clause_attention(clause_features).squeeze(
                -1
            )  # [num_clauses]

            # Apply softmax for normalization
            var_weights = F.softmax(var_weights, dim=0)  # [num_variables]
            clause_weights = F.softmax(clause_weights, dim=0)  # [num_clauses]

            # Apply attention
            var_pooled = torch.sum(
                var_embeddings * var_weights.unsqueeze(1), dim=0
            )  # [output_dim]
            clause_pooled = torch.sum(
                clause_embeddings * clause_weights.unsqueeze(1), dim=0
            )  # [output_dim]

            graph_embedding = var_pooled + clause_pooled  # [output_dim]

        return {
            "graph_embedding": graph_embedding,
            "variable_embeddings": var_embeddings,
            "clause_embeddings": clause_embeddings,
        }


class GNNObservationEncoder:
    """
    Encodes SAT problems as graph neural networks for RL agents.
    Compatible with the ModularObservationPreprocessor interface.
    """

    def __init__(
        self,
        num_variables: int,
        num_clauses: int,
        embedding_dim: int = 128,
        message_passing_steps: int = 3,
        include_raw_features: bool = True,
        device: str = "cpu",
    ):
        """
        Initialize the GNN Observation Encoder.

        Args:
            num_variables: Number of variables in SAT problems
            num_clauses: Number of clauses in SAT problems
            embedding_dim: Dimensionality of the output embeddings
            message_passing_steps: Number of message passing steps in the GNN
            include_raw_features: Whether to include raw variable/clause features alongside embeddings
            device: Device to run the GNN on ('cpu' or 'cuda')
        """
        self.num_variables = num_variables
        self.num_clauses = num_clauses
        self.embedding_dim = embedding_dim
        self.include_raw_features = include_raw_features
        self.device = device

        # Create the GNN model
        self.gnn = SATGraphNN(
            num_variables=num_variables,
            num_clauses=num_clauses,
            var_feature_dim=3,  # [unassigned, false, true]
            clause_feature_dim=3,  # [unsatisfied, satisfied, unknown]
            message_dim=64,
            num_message_passing_steps=message_passing_steps,
            output_dim=embedding_dim,
            graph_pooling="attention",
        ).to(device)

        # Graph builder for creating graph inputs
        self.graph_builder = GraphBuilder(num_variables, num_clauses)

    def encode(self, sat_state: dict) -> dict[str, np.ndarray | torch.Tensor]:
        """
        Encode a SAT state into graph embeddings.

        Args:
            sat_state: Dictionary with SAT problem state:
                - 'clauses': List of clauses
                - 'variable_assignments': Current variable assignments

        Returns:
            Dictionary with encoded representations:
                - 'graph_embedding': Global graph representation
                - 'variable_embeddings': Per-variable embeddings
                - 'clause_embeddings': Per-clause embeddings
                - 'variable_features': Raw variable features (if include_raw_features=True)
                - 'clause_features': Raw clause features (if include_raw_features=True)
        """
        # Prepare the problem for the GNN
        problem = {
            "clauses": sat_state["clauses"],
            "variable_assignments": sat_state.get("variable_assignments", {}),
        }

        # Forward pass through the GNN
        with torch.no_grad():
            gnn_output = self.gnn(problem)

        # Convert to numpy arrays for the encoder interface
        result = {
            "graph_embedding": gnn_output["graph_embedding"].cpu().numpy(),
            "variable_embeddings": gnn_output["variable_embeddings"].cpu().numpy(),
            "clause_embeddings": gnn_output["clause_embeddings"].cpu().numpy(),
        }

        # Include raw features if requested
        if self.include_raw_features:
            graph_data = self.graph_builder.build_bipartite_graph(
                sat_state["clauses"], sat_state.get("variable_assignments", {})
            )

            result["variable_features"] = graph_data["variable_features"].numpy()
            result["clause_features"] = graph_data["clause_features"].numpy()

        return result

    def get_observation_space(self) -> dict:
        """
        Get the observation space specification.

        Returns:
            Dictionary describing the observation space structure and shapes
        """
        obs_space = {
            "graph_embedding": {"shape": (self.embedding_dim,), "dtype": np.float32},
            "variable_embeddings": {
                "shape": (self.num_variables, self.embedding_dim),
                "dtype": np.float32,
            },
            "clause_embeddings": {
                "shape": (self.num_clauses, self.embedding_dim),
                "dtype": np.float32,
            },
        }

        if self.include_raw_features:
            obs_space["variable_features"] = {
                "shape": (self.num_variables, 3),
                "dtype": np.float32,
            }
            obs_space["clause_features"] = {
                "shape": (self.num_clauses, 3),
                "dtype": np.float32,
            }

        return obs_space

    def train(self, mode: bool = True):
        """
        Set the GNN to training mode.

        Args:
            mode: Whether to set training mode (True) or evaluation mode (False)
        """
        self.gnn.train(mode)

    def eval(self):
        """Set the GNN to evaluation mode."""
        self.gnn.eval()

    def to(self, device: str):
        """
        Move the GNN to the specified device.

        Args:
            device: Device to move the GNN to ('cpu' or 'cuda')
        """
        self.device = device
        self.gnn.to(device)
        return self
