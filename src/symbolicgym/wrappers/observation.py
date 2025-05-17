"""Observation wrappers for converting between dict, flat, and graph formats."""

import numpy as np


class ObservationWrapper:
    def __init__(self, mode="dict"):
        self.mode = mode

    def convert(self, obs):
        if self.mode == "dict":
            return obs
        elif self.mode == "flat":
            # Flatten all arrays in dict to a single vector
            if isinstance(obs, dict):
                flat = []
                for v in obs.values():
                    if isinstance(v, dict):
                        flat.extend(list(v.values()))
                    elif isinstance(v, np.ndarray):
                        flat.extend(v.flatten())
                    else:
                        flat.append(v)
                return np.array(flat, dtype=np.float32)
            return obs
        elif self.mode == "graph":
            # Convert dict obs to graph data for GNNs (PyG format)
            # Expects obs to have 'nodes' and 'edges' keys or convert from variable/clause structure
            if isinstance(obs, dict) and "nodes" in obs and "edges" in obs:
                # Already in graph format
                return obs
            elif isinstance(obs, dict) and "variables" in obs and "clauses" in obs:
                # Convert variable/clause dict to graph
                nodes = np.array(obs["variables"]).reshape(-1, 1)
                # Build edges: connect variables to clauses (bipartite)
                num_vars = len(obs["variables"])
                num_clauses = len(obs["clauses"])
                edges = []
                for ci in range(num_clauses):
                    # For SAT, connect all variables to each clause (or use clause structure if available)
                    for vi in range(num_vars):
                        edges.append((vi, num_vars + ci))  # variable to clause node
                        edges.append(
                            (num_vars + ci, vi)
                        )  # clause to variable node (undirected)
                edge_index = np.array(edges).T  # shape (2, num_edges)
                return {"nodes": nodes, "edges": edge_index}
            else:
                # For non-SAT domains, return a trivial graph
                nodes = np.array([1.0]).reshape(-1, 1)
                edge_index = np.zeros((2, 0), dtype=int)
                return {"nodes": nodes, "edges": edge_index}
        else:
            raise ValueError(f"Unknown observation mode: {self.mode}")
