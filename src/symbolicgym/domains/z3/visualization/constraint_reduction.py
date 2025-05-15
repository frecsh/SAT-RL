"""
Constraint reduction visualizer for Z3 domain.
"""
import matplotlib.pyplot as plt


def plot_constraint_reduction(initial_constraints, final_constraints, ax=None):
    """
    Plot reduction in number of constraints before and after solving.
    Args:
        initial_constraints: list
        final_constraints: list
        ax: matplotlib axis (optional)
    """
    if ax is None:
        fig, ax = plt.subplots()
    ax.bar(
        ["Initial", "Final"],
        [len(initial_constraints), len(final_constraints)],
        color=["red", "green"],
    )
    ax.set_ylabel("Number of Constraints")
    ax.set_title("Constraint Reduction")
    return ax
