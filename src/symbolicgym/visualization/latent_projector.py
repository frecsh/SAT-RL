"""
Cross-domain latent space projector for SymbolicGym.
"""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP


def project_latent_space(X, method="pca", n_components=2, **kwargs):
    """
    Project latent space using PCA, t-SNE, or UMAP.
    Args:
        X: np.ndarray
        method: 'pca', 'tsne', or 'umap'
        n_components: int
        **kwargs: additional arguments for the projection method
    Returns:
        X_proj: np.ndarray
    """
    if method == "pca":
        return PCA(n_components=n_components, **kwargs).fit_transform(X)
    elif method == "tsne":
        return TSNE(n_components=n_components, **kwargs).fit_transform(X)
    elif method == "umap":
        return UMAP(n_components=n_components, **kwargs).fit_transform(X)
    else:
        raise ValueError(f"Unknown method: {method}")
