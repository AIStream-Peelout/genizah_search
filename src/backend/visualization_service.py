"""
Visualization Service - Proper dimensionality reduction usin standard Python libraries

This service provides PCA, t-SNE, and UMAP implementations using standard
scientific Python libraries (scikit-learn, umap-learn) instead of JavaScript...
"""

import numpy as np
from typing import List, Dict, Any, Optional
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import logging

logger = logging.getLogger(__name__)


class VisualizationService:
    """Service for performing dimensionality reduction on embeddings"""
    
    @staticmethod
    def perform_pca(
        embeddings: List[List[float]], 
        n_components: int = 2,
        random_state: Optional[int] = 42
    ) -> List[List[float]]:
        """
        Perform Principal Component Analysis (PCA) on embeddings.
        
        Args:
            embeddings: List of embedding vectors
            n_components: Number of dimensions to reduce to (default: 2)
            random_state: Random seed for reproducibility
            
        Returns:
            List of 2D coordinates
        """
        if len(embeddings) < 2:
            return [[0.0, 0.0] for _ in embeddings]
        
        # Convert to numpy array
        X = np.array(embeddings)
        
        # Ensure we don't request more components than we have samples
        n_components = min(n_components, X.shape[0] - 1, X.shape[1])
        
        logger.info(f"Performing PCA on {len(embeddings)} embeddings of dimension {X.shape[1]}, reducing to {n_components}D")
        
        # Perform PCA
        pca = PCA(n_components=n_components, random_state=random_state)
        coords = pca.fit_transform(X)
        
        # Convert back to list format
        result = coords.tolist()
        
        logger.info(f"PCA explained variance ratio: {pca.explained_variance_ratio_.sum():.4f}")
        
        return result
    
    @staticmethod
    def perform_tsne(
        embeddings: List[List[float]],
        perplexity: Optional[int] = None,
        n_iter: int = 1000,
        learning_rate: float = 200.0,
        random_state: Optional[int] = 42,
        early_exaggeration: float = 12.0
    ) -> List[List[float]]:
        """
        Perform t-SNE (t-Distributed Stochastic Neighbor Embedding) on embeddings.
        
        Args:
            embeddings: List of embedding vectors
            perplexity: Perplexity parameter (default: min(30, n_samples-1))
            n_iter: Number of iterations (default: 1000) - mapped to max_iter for scikit-learn
            learning_rate: Learning rate (default: 200.0)
            random_state: Random seed for reproducibility
            early_exaggeration: Early exaggeration factor (default: 12.0)
            
        Returns:
            List of 2D coordinates
        """
        if len(embeddings) < 2:
            return [[0.0, 0.0] for _ in embeddings]
        
        # Convert to numpy array
        X = np.array(embeddings)
        n_samples = X.shape[0]
        
        # Set default perplexity if not provided
        if perplexity is None:
            perplexity = min(30, max(5, (n_samples - 1) // 3))
        
        # Ensure perplexity is valid
        perplexity = min(perplexity, n_samples - 1)
        
        logger.info(f"Performing t-SNE on {n_samples} embeddings of dimension {X.shape[1]}, "
                   f"perplexity={perplexity}, iterations={n_iter}")
        
        # Perform t-SNE
        # Note: scikit-learn uses 'max_iter' instead of 'n_iter'
        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            max_iter=n_iter,  # Use max_iter for scikit-learn compatibility
            learning_rate=learning_rate,
            random_state=random_state,
            early_exaggeration=early_exaggeration,
            verbose=1 if n_samples > 100 else 0
        )
        
        coords = tsne.fit_transform(X)
        
        # Convert back to list format
        result = coords.tolist()
        
        logger.info(f"t-SNE completed successfully")
        
        return result
    
    @staticmethod
    def perform_umap(
        embeddings: List[List[float]],
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        n_components: int = 2,
        metric: str = 'cosine',
        random_state: Optional[int] = 42
    ) -> List[List[float]]:
        """
        Perform UMAP (Uniform Manifold Approximation and Projection) on embeddings.
        
        Args:
            embeddings: List of embedding vectors
            n_neighbors: Number of neighbors (default: 15)
            min_dist: Minimum distance between points in embedding space (default: 0.1)
            n_components: Number of dimensions to reduce to (default: 2)
            metric: Distance metric to use (default: 'cosine')
            random_state: Random seed for reproducibility
            
        Returns:
            List of 2D coordinates
        """
        if len(embeddings) < 2:
            return [[0.0, 0.0] for _ in embeddings]
        
        # Convert to numpy array
        X = np.array(embeddings)
        n_samples = X.shape[0]
        
        # Adjust n_neighbors if needed
        n_neighbors = min(n_neighbors, n_samples - 1)
        n_neighbors = max(n_neighbors, 2)  # At least 2 neighbors
        
        logger.info(f"Performing UMAP on {n_samples} embeddings of dimension {X.shape[1]}, "
                   f"n_neighbors={n_neighbors}, min_dist={min_dist}")
        
        # Perform UMAP
        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=n_components,
            metric=metric,
            random_state=random_state,
            verbose=True if n_samples > 100 else False
        )
        
        coords = reducer.fit_transform(X)
        
        # Convert back to list format
        result = coords.tolist()
        
        logger.info(f"UMAP completed successfully")
        
        return result
    
    @staticmethod
    def calculate_visualization(
        embeddings: List[List[float]],
        method: str = 'tsne',
        **kwargs
    ) -> Dict[str, Any]:
        """
        Calculate visualization coordinates using the specified method.
        
        Args:
            embeddings: List of embedding vectors
            method: Method to use ('pca', 'tsne', or 'umap')
            **kwargs: Additional parameters for the specific method
            
        Returns:
            Dictionary with coordinates and metadata
        """
        if not embeddings or len(embeddings) == 0:
            raise ValueError("No embeddings provided")
        
        method = method.lower()
        
        if method == 'pca':
            coords = VisualizationService.perform_pca(
                embeddings,
                n_components=kwargs.get('n_components', 2),
                random_state=kwargs.get('random_state', 42)
            )
        elif method == 'tsne':
            coords = VisualizationService.perform_tsne(
                embeddings,
                perplexity=kwargs.get('perplexity'),
                n_iter=kwargs.get('n_iter', 1000),
                learning_rate=kwargs.get('learning_rate', 200.0),
                random_state=kwargs.get('random_state', 42),
                early_exaggeration=kwargs.get('early_exaggeration', 12.0)
            )
        elif method == 'umap':
            coords = VisualizationService.perform_umap(
                embeddings,
                n_neighbors=kwargs.get('n_neighbors', 15),
                min_dist=kwargs.get('min_dist', 0.1),
                n_components=kwargs.get('n_components', 2),
                metric=kwargs.get('metric', 'cosine'),
                random_state=kwargs.get('random_state', 42)
            )
        else:
            raise ValueError(f"Unknown visualization method: {method}. Must be 'pca', 'tsne', or 'umap'")
        
        # Calculate statistics
        coords_array = np.array(coords)
        x_min, x_max = float(coords_array[:, 0].min()), float(coords_array[:, 0].max())
        y_min, y_max = float(coords_array[:, 1].min()), float(coords_array[:, 1].max())
        
        return {
            'coordinates': coords,
            'method': method,
            'num_points': len(coords),
            'statistics': {
                'x_range': [x_min, x_max],
                'y_range': [y_min, y_max]
            }
        }


# Create singleton instance
visualization_service = VisualizationService()

