"""
Action entropy visualization.

This module provides tools for visualizing the entropy of agent actions over time,
which can help in understanding exploration patterns.
"""


import matplotlib.pyplot as plt
import numpy as np


class ActionEntropyPlotter:
    """
    A tool for plotting action entropy over time.

    Attributes:
        num_actions: Number of possible actions
        actions: List of taken actions
        window_size: Size of the sliding window for entropy calculation
    """

    def __init__(self, num_actions: int, window_size: int = 20):
        """
        Initialize the action entropy plotter.

        Args:
            num_actions: Number of possible actions
            window_size: Size of the sliding window for entropy calculation
        """
        self.num_actions = num_actions
        self.window_size = window_size
        self.actions = []

    def add_action(self, action: int):
        """
        Add an action to the record.

        Args:
            action: The action taken by the agent

        Raises:
            ValueError: If the action is invalid
        """
        if action < 0 or action >= self.num_actions:
            raise ValueError(
                f"Invalid action {action}. Must be in range [0, {self.num_actions-1}]"
            )

        self.actions.append(action)

    def calculate_entropy(self, actions: list[int] | None = None) -> float:
        """
        Calculate the entropy of the action distribution.

        Args:
            actions: List of actions to calculate entropy for, or None to use all recorded actions

        Returns:
            The entropy value

        Notes:
            Entropy is calculated as -sum(p*log(p)) where p is the probability of each action
        """
        if actions is None:
            actions = self.actions

        if not actions:
            return 0.0

        # Count occurrences of each action
        counts = np.zeros(self.num_actions)
        for a in actions:
            counts[a] += 1

        # Calculate probabilities
        probs = counts / len(actions)

        # Calculate entropy (-sum(p*log(p)))
        # Use only non-zero probabilities to avoid log(0)
        entropy = 0.0
        for p in probs:
            if p > 0:
                entropy -= p * np.log(p)

        return entropy

    def calculate_entropies(self) -> list[float]:
        """
        Calculate entropy for sliding windows of actions.

        Returns:
            List of entropy values for each window
        """
        if len(self.actions) < self.window_size:
            return [self.calculate_entropy()]

        entropies = []
        for i in range(len(self.actions) - self.window_size + 1):
            window = self.actions[i : i + self.window_size]
            entropies.append(self.calculate_entropy(window))

        return entropies

    def plot_entropy(
        self, ax: plt.Axes | None = None, running_window: bool = True, **kwargs
    ) -> plt.Axes:
        """
        Plot the entropy of actions over time.

        Args:
            ax: Matplotlib axes to draw on
            running_window: Whether to use a running window (True) or entire history (False)
            **kwargs: Additional kwargs to pass to plot

        Returns:
            The matplotlib axes with the plotted entropy
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        if not self.actions:
            ax.text(0.5, 0.5, "No actions recorded", ha="center", va="center")
            return ax

        if running_window:
            entropies = self.calculate_entropies()
            x = list(range(len(entropies)))
            window_label = f" (window size: {self.window_size})"
        else:
            # Calculate cumulative entropies
            entropies = []
            for i in range(1, len(self.actions) + 1):
                entropies.append(self.calculate_entropy(self.actions[:i]))
            x = list(range(len(entropies)))
            window_label = ""

        ax.plot(x, entropies, **kwargs)
        ax.set_xlabel("Step")
        ax.set_ylabel("Entropy")
        ax.set_title(f"Action Entropy Over Time{window_label}")
        ax.grid(True, linestyle="--", alpha=0.7)

        # Add a reference line for maximum entropy
        max_entropy = np.log(self.num_actions)
        ax.axhline(
            max_entropy,
            color="red",
            linestyle="--",
            alpha=0.5,
            label=f"Max Entropy ({max_entropy:.2f})",
        )

        ax.legend()

        return ax
