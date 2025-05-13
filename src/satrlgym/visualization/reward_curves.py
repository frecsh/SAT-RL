"""
Reward curve visualization.

This module provides tools for visualizing reward curves and other metrics
across episodes and runs.
"""


import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes


class RewardCurvePlotter:
    """
    A tool for plotting reward curves and satisfaction progress.

    Attributes:
        episodes: List of episode numbers
        rewards: List of reward sequences for each episode
        satisfied_clauses: List of satisfied clause counts for each episode
        total_clauses: List of total clause counts for each episode
    """

    def __init__(self):
        """
        Initialize the reward curve plotter.
        """
        self.episodes = []
        self.rewards = []
        self.satisfied_clauses = []
        self.total_clauses = []

    def add_episode(
        self,
        episode_rewards: list[float],
        episode_satisfied: list[int],
        total_clauses: int,
    ):
        """
        Add an episode's data to the plotter.

        Args:
            episode_rewards: List of rewards for each step in the episode
            episode_satisfied: List of satisfied clause counts for each step
            total_clauses: Total number of clauses in the problem
        """
        self.episodes.append(len(self.episodes))
        self.rewards.append(episode_rewards)
        self.satisfied_clauses.append(episode_satisfied)
        self.total_clauses.append(total_clauses)

    def plot_rewards(
        self,
        ax: Axes | None = None,
        show_individual: bool = True,
        show_mean: bool = True,
        **kwargs,
    ) -> Axes:
        """
        Plot the reward curves.

        Args:
            ax: Matplotlib axes to draw on
            show_individual: Whether to show individual episode curves
            show_mean: Whether to show the mean reward curve
            **kwargs: Additional kwargs to pass to plot

        Returns:
            The matplotlib axes with the plotted curves
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        if not self.rewards:
            ax.text(0.5, 0.5, "No data available", ha="center", va="center")
            return ax

        # Plot individual episode rewards
        if show_individual and len(self.rewards) > 0:
            for i, rewards in enumerate(self.rewards):
                x = list(range(len(rewards)))
                ax.plot(
                    x,
                    rewards,
                    alpha=0.3,
                    label=f"Episode {i}" if i == 0 else None,
                    **kwargs,
                )

        # Plot mean reward
        if show_mean and len(self.rewards) > 0:
            # Determine max length
            max_len = max(len(r) for r in self.rewards)

            # Pad shorter sequences
            padded_rewards = []
            for rewards in self.rewards:
                if len(rewards) < max_len:
                    # Pad with the last value
                    padded = rewards + [rewards[-1]] * (max_len - len(rewards))
                    padded_rewards.append(padded)
                else:
                    padded_rewards.append(rewards)

            # Calculate mean
            mean_rewards = np.mean(padded_rewards, axis=0)
            x = list(range(len(mean_rewards)))

            # Plot mean
            ax.plot(
                x,
                mean_rewards,
                linewidth=2,
                color="black",
                label="Mean Reward",
                **kwargs,
            )

        ax.set_xlabel("Step")
        ax.set_ylabel("Reward")
        ax.set_title("Reward Curves")
        ax.grid(True, linestyle="--", alpha=0.7)

        if show_individual or show_mean:
            ax.legend()

        return ax

    def plot_satisfied_clauses(
        self,
        ax: Axes | None = None,
        normalized: bool = True,
        show_individual: bool = True,
        show_mean: bool = True,
        **kwargs,
    ) -> Axes:
        """
        Plot the satisfied clauses curves.

        Args:
            ax: Matplotlib axes to draw on
            normalized: Whether to normalize by the total number of clauses
            show_individual: Whether to show individual episode curves
            show_mean: Whether to show the mean curve
            **kwargs: Additional kwargs to pass to plot

        Returns:
            The matplotlib axes with the plotted curves
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        if not self.satisfied_clauses:
            ax.text(0.5, 0.5, "No data available", ha="center", va="center")
            return ax

        # Plot individual episode curves
        if show_individual and len(self.satisfied_clauses) > 0:
            for i, satisfied in enumerate(self.satisfied_clauses):
                x = list(range(len(satisfied)))
                if normalized:
                    y = [s / self.total_clauses[i] for s in satisfied]
                    ylabel = "Satisfaction Ratio"
                else:
                    y = satisfied
                    ylabel = "Satisfied Clauses"

                ax.plot(
                    x, y, alpha=0.3, label=f"Episode {i}" if i == 0 else None, **kwargs
                )

        # Plot mean curve
        if show_mean and len(self.satisfied_clauses) > 0:
            # Determine max length
            max_len = max(len(s) for s in self.satisfied_clauses)

            # Pad shorter sequences
            padded_satisfied = []
            for i, satisfied in enumerate(self.satisfied_clauses):
                if len(satisfied) < max_len:
                    # Pad with the last value
                    padded = satisfied + [satisfied[-1]] * (max_len - len(satisfied))
                    padded_satisfied.append(padded)
                else:
                    padded_satisfied.append(satisfied)

            # Normalize if needed
            if normalized:
                padded_norm = []
                for i, satisfied in enumerate(padded_satisfied):
                    norm = [s / self.total_clauses[i] for s in satisfied]
                    padded_norm.append(norm)

                # Calculate mean
                mean_satisfied = np.mean(padded_norm, axis=0)
                ylabel = "Satisfaction Ratio"
            else:
                # Calculate mean
                mean_satisfied = np.mean(padded_satisfied, axis=0)
                ylabel = "Satisfied Clauses"

            x = list(range(len(mean_satisfied)))

            # Plot mean
            ax.plot(
                x, mean_satisfied, linewidth=2, color="black", label="Mean", **kwargs
            )

        ax.set_xlabel("Step")
        ax.set_ylabel(ylabel)
        ax.set_title("Clause Satisfaction Progress")
        ax.grid(True, linestyle="--", alpha=0.7)

        if normalized:
            ax.set_ylim(0, 1)

        if show_individual or show_mean:
            ax.legend()

        return ax
