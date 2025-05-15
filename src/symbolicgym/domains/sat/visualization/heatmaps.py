"""
SAT variable-flip heatmap visualization for SymbolicGym.
"""
import matplotlib.pyplot as plt
import numpy as np


def plot_variable_flip_heatmap(flip_counts, var_names=None, ax=None):
    """
    Plot a heatmap of variable flip counts.
    Args:
        flip_counts: list or np.ndarray of counts per variable
        var_names: list of variable names (optional)
        ax: matplotlib axis (optional)
    """
    flip_counts = np.array(flip_counts)
    if ax is None:
        fig, ax = plt.subplots()
    im = ax.imshow(flip_counts[None, :], aspect="auto", cmap="hot")
    ax.set_yticks([])
    if var_names:
        ax.set_xticks(np.arange(len(var_names)))
        ax.set_xticklabels(var_names)
    else:
        ax.set_xticks(np.arange(len(flip_counts)))
    ax.set_title("Variable Flip Heatmap")
    plt.colorbar(im, ax=ax)
    return ax
