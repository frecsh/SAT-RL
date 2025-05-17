"""Dimensionality reduction and visualization utilities for symbolic feedback vectors.

This module provides tools for reducing high-dimensional feedback vectors to
lower-dimensional representations for visualization and analysis.
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA, TSNE
from sklearn.manifold import UMAP

from symbolicgym.core.feedback_vectors import SymbolicFeedbackVector


class FeedbackDimensionReducer:
    """Reduces dimensionality of feedback vectors for visualization."""

    def __init__(self, method: str = "pca", n_components: int = 2):
        """Initialize dimension reducer.

        Args:
            method: Reduction method ('pca', 'tsne', or 'umap')
            n_components: Number of dimensions to reduce to
        """
        self.method = method.lower()
        self.n_components = n_components
        self.reducer = self._create_reducer()
        self.is_fitted = False

    def _create_reducer(self):
        """Create the appropriate dimensionality reduction object."""
        if self.method == "pca":
            return PCA(n_components=self.n_components)
        elif self.method == "tsne":
            return TSNE(n_components=self.n_components)
        elif self.method == "umap":
            return UMAP(n_components=self.n_components)
        else:
            raise ValueError(f"Unknown reduction method: {self.method}")

    def fit_transform(self, vectors: list[SymbolicFeedbackVector]) -> np.ndarray:
        """Fit the reducer and transform the vectors.

        Args:
            vectors: List of feedback vectors to reduce

        Returns:
            Array of reduced vectors
        """
        data = np.array([v.as_array() for v in vectors])
        reduced = self.reducer.fit_transform(data)
        self.is_fitted = True
        return reduced

    def transform(self, vectors: list[SymbolicFeedbackVector]) -> np.ndarray:
        """Transform new vectors using previously fit reducer.

        Args:
            vectors: List of feedback vectors to reduce

        Returns:
            Array of reduced vectors
        """
        if not self.is_fitted:
            raise RuntimeError("Reducer must be fitted before transform")
        data = np.array([v.as_array() for v in vectors])
        return self.reducer.transform(data)

    def plot_reduction(
        self,
        vectors: list[SymbolicFeedbackVector],
        labels: list[str] | None = None,
        title: str | None = None,
    ) -> plt.Figure:
        """Create a scatter plot of reduced vectors.

        Args:
            vectors: List of feedback vectors to plot
            labels: Optional list of labels for each point
            title: Optional plot title

        Returns:
            Matplotlib figure
        """
        reduced = (
            self.fit_transform(vectors)
            if not self.is_fitted
            else self.transform(vectors)
        )

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.scatter(reduced[:, 0], reduced[:, 1])

        if labels:
            for i, label in enumerate(labels):
                ax.annotate(label, (reduced[i, 0], reduced[i, 1]))

        if title:
            ax.set_title(title)
        else:
            ax.set_title(f"{self.method.upper()} Reduction of Feedback Vectors")

        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")

        return fig


def create_trajectory_animation(
    vectors: list[SymbolicFeedbackVector],
    labels: list[str] | None = None,
    title: str | None = None,
) -> plt.Figure:
    """Create an animated trajectory of feedback vectors over time.

    Args:
        vectors: List of feedback vectors in temporal order
        labels: Optional list of labels for trajectory points
        title: Optional animation title

    Returns:
        Matplotlib figure with animation
    """
    # TODO: Implement trajectory animation using matplotlib.animation
    raise NotImplementedError
