"""Step importance heatmap for SymPy domain."""

import matplotlib.pyplot as plt
import numpy as np


def plot_step_importance_heatmap(step_importances, ax=None):
    """Plot a heatmap of step importances.

    Args:
        step_importances: list or np.ndarray of importances
        ax: matplotlib axis (optional)
    """
    step_importances = np.array(step_importances)
    if ax is None:
        fig, ax = plt.subplots()
    im = ax.imshow(step_importances[None, :], aspect="auto", cmap="viridis")
    ax.set_yticks([])
    ax.set_xticks(np.arange(len(step_importances)))
    ax.set_title("Step Importance Heatmap")
    plt.colorbar(im, ax=ax)
    return ax
