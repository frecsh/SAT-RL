"""
Variable assignment timeline visualization.

This module provides a visualization of variable assignments over time
in the form of a color-coded timeline.
"""


import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


class AssignmentTimeline:
    """
    Timeline visualization of variable assignments.

    Attributes:
        num_variables: Number of variables in the problem
        max_steps: Maximum number of steps to track
        data: 2D array of assignment values over time
    """

    def __init__(self, num_variables: int, max_steps: int):
        """
        Initialize the timeline.

        Args:
            num_variables: Number of variables in the problem
            max_steps: Maximum number of steps to track
        """
        self.num_variables = num_variables
        self.max_steps = max_steps

        # Initialize data array with NaN (no assignment)
        self.data = np.full((max_steps, num_variables), np.nan)

    def update(self, step: int, assignment: dict[int, bool]):
        """
        Update the timeline with a new assignment at a given step.

        Args:
            step: The step number (0-indexed)
            assignment: Dictionary mapping variable indices (1-indexed) to Boolean values

        Raises:
            IndexError: If step is out of range
        """
        if step < 0 or step >= self.max_steps:
            raise IndexError(f"Step {step} is out of range [0, {self.max_steps-1}]")

        # Convert to 0-indexed for the array
        for var, value in assignment.items():
            var_idx = var - 1
            if 0 <= var_idx < self.num_variables:
                self.data[step, var_idx] = float(value)

    def plot(self, ax=None, **kwargs):
        """
        Plot the assignment timeline.

        Args:
            ax: Matplotlib axes to draw on
            **kwargs: Additional kwargs to pass to imshow

        Returns:
            The matplotlib axes with the plotted timeline
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))

        # Create a custom colormap for True (green), False (red), and unassigned (gray)
        cmap = ListedColormap(["red", "green", "gray"])

        # Clone data to avoid modifying the original
        plot_data = self.data.copy()

        # Replace NaN with 2 (for gray color)
        plot_data = np.where(np.isnan(plot_data), 2, plot_data)

        # Plot the data
        im = ax.imshow(
            plot_data.T,  # Transpose to have variables on Y-axis
            aspect="auto",
            interpolation="nearest",
            cmap=cmap,
            **kwargs,
        )

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, ticks=[0, 1, 2])
        cbar.ax.set_yticklabels(["False", "True", "Unassigned"])

        # Set labels
        ax.set_xlabel("Step")
        ax.set_ylabel("Variable")
        ax.set_title("Variable Assignment Timeline")

        # Customize y-axis to show variable numbers (1-indexed)
        ax.set_yticks(np.arange(self.num_variables))
        ax.set_yticklabels([f"Var {i+1}" for i in range(self.num_variables)])

        return ax
