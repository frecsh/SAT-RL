import matplotlib.pyplot as plt
import numpy as np

from symbolicgym.analysis.feedback_interpreter import interpret_feedback_vector
from symbolicgym.domains.sat.visualization.attention_rollout import attention_rollout
from symbolicgym.domains.sat.visualization.heatmaps import plot_variable_flip_heatmap
from symbolicgym.domains.sat.visualization.proof_analyzer import analyze_drat_proof
from symbolicgym.domains.sympy.visualization.step_importance import (
    plot_step_importance_heatmap,
)
from symbolicgym.domains.sympy.visualization.transformation_graphs import (
    plot_transformation_graph,
)
from symbolicgym.domains.z3.visualization.constraint_reduction import (
    plot_constraint_reduction,
)
from symbolicgym.visualization.dashboard import show_dashboard
from symbolicgym.visualization.latent_projector import project_latent_space

# SAT variable-flip heatmap
flip_counts = np.random.randint(0, 10, size=8)
plot_variable_flip_heatmap(flip_counts, var_names=[f"x{i+1}" for i in range(8)])

# SAT GNN attention rollout
attn_weights = [np.random.rand(8, 8) for _ in range(3)]
rollout = attention_rollout(attn_weights)
plt.imshow(rollout, cmap="viridis")
plt.title("GNN Attention Rollout")
plt.colorbar()
plt.show()

# SAT DRAT proof analyzer
proof_stats = analyze_drat_proof(["step1", "step2", "step3"])
print("DRAT proof stats:", proof_stats)

# SymPy transformation graph
transformations = [("x+1", "x+2", "add 1"), ("x+2", "2*x+2", "expand")]
plot_transformation_graph(transformations)

# SymPy step importance heatmap
importances = np.random.rand(5)
plot_step_importance_heatmap(importances)

# Z3 constraint reduction
plot_constraint_reduction([1, 2, 3, 4, 5], [1, 2, 3])

# Latent space projection
latents = np.random.rand(100, 16)
labels = np.random.choice(["SAT", "SymPy", "Z3"], size=100)
proj = project_latent_space(latents, method="pca")
plt.scatter(
    proj[:, 0],
    proj[:, 1],
    c=[{"SAT": 0, "SymPy": 1, "Z3": 2}[l] for l in labels],
    cmap="tab10",
)
plt.title("Latent Space Projection (PCA)")
plt.show()

# Feedback vector interpretation
feedback = np.random.randn(8)
interpretation = interpret_feedback_vector(feedback, domain="sat")
print("Feedback interpretation:", interpretation)

# Unified dashboard (example: combine 2 plots)
fig1, ax1 = plt.subplots()
plot_variable_flip_heatmap(flip_counts, var_names=[f"x{i+1}" for i in range(8)], ax=ax1)
fig2, ax2 = plt.subplots()
plot_step_importance_heatmap(importances, ax=ax2)
show_dashboard([fig1, fig2], layout=(1, 2))
