"""Example usage of SharedEncoder with advanced GNNs in a training pipeline."""

import torch

from symbolicgym.models.shared_encoder import SharedEncoder
from symbolicgym.wrappers.observation import ObservationWrapper

# Example: using DeeperGNN encoder
obs_dict = {
    "variables": [1, -1, 1],
    "clauses": [1, 0, 1],
}
obs_wrapper = ObservationWrapper(mode="graph")
graph_obs = obs_wrapper.convert(obs_dict)
# graph_obs: {"nodes": ..., "edges": ...}

# Convert numpy arrays to torch tensors for GNN
num_vars = len(obs_dict["variables"])
num_clauses = len(obs_dict["clauses"])
# Stack variable and clause nodes
nodes = torch.tensor(
    [[v] for v in obs_dict["variables"]] + [[c] for c in obs_dict["clauses"]],
    dtype=torch.float32,
)
edges = torch.tensor(graph_obs["edges"], dtype=torch.long)
graph_data = {"nodes": nodes, "edges": edges}

# Instantiate encoder (choose encoder_type: "deeper_gnn", "gat", "edge_mp", etc.)
encoder = SharedEncoder(input_dim=1, embed_dim=16, encoder_type="deeper_gnn")

# Forward pass
embedding = encoder(graph_data)
print("Graph embedding shape:", embedding.shape)

# --- Integration in a training loop ---
# Assume you have a batch of graph observations from the environment
# for obs in batch:
#     graph_obs = obs_wrapper.convert(obs)
#     nodes = torch.tensor(graph_obs["nodes"], dtype=torch.float32)
#     edges = torch.tensor(graph_obs["edges"], dtype=torch.long)
#     graph_data = {"nodes": nodes, "edges": edges}
#     embedding = encoder(graph_data)
#     # Use embedding as input to policy/value network, loss computation, etc.
