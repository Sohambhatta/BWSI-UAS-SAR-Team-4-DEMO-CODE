import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from typing import Tuple, Optional

class SARSpatialOptimizer:
    """
    Global spatial optimization for SAR sampling uniformity across multiple passes.
    """
    
    def __init__(self, coordinates: np.ndarray, amplitudes: np.ndarray):
        """
        Initialize with coordinate and amplitude data.
        
        Args:
            coordinates: Nx3 array of [x, y, z] positions
            amplitudes: Nx1 array of corresponding range-amplitude values
        """
        self.coordinates = coordinates.copy()
        self.amplitudes = amplitudes.copy()
        self.n_samples = len(coordinates)
        
    def spatial_binning_filter(self, bin_size: float, 
                              selection_method: str = 'center') -> Tuple[np.ndarray, np.ndarray]:
        """
        Divide space into bins and select one sample per bin.
        
        Args:
            bin_size: Size of spatial bins (in same units as coordinates)
            selection_method: 'center' or 'random'
        
        Returns:
            filtered_coords, filtered_amplitudes
        """
        # Create spatial bins
        min_coords = np.min(self.coordinates, axis=0)
        max_coords = np.max(self.coordinates, axis=0)
        
        # Calculate bin indices for each point
        bin_indices = np.floor((self.coordinates - min_coords) / bin_size).astype(int)
        
        # Create unique bin identifiers
        bin_ids = np.array([tuple(idx) for idx in bin_indices])
        unique_bins = np.unique(bin_ids)
        
        selected_indices = []
        
        for bin_id in unique_bins:
            # Find all points in this bin
            mask = np.array([tuple(idx) == bin_id for idx in bin_indices])
            bin_points = np.where(mask)[0]
            
            if len(bin_points) == 1:
                selected_indices.append(bin_points[0])
            else:
                # Select based on method
                if selection_method == 'center':
                    # Find point closest to bin center
                    bin_coords = self.coordinates[bin_points]
                    bin_center = np.mean(bin_coords, axis=0)
                    distances = np.linalg.norm(bin_coords - bin_center, axis=1)
                    best_idx = bin_points[np.argmin(distances)]
                elif selection_method == 'random':
                    best_idx = np.random.choice(bin_points)
                else:
                    raise ValueError(f"Unknown selection method: {selection_method}")
                
                selected_indices.append(best_idx)
        
        selected_indices = np.array(selected_indices)
        return self.coordinates[selected_indices], self.amplitudes[selected_indices]
    
    def distance_based_filter(self, min_distance: float, 
                             selection_method: str = 'random') -> Tuple[np.ndarray, np.ndarray]:
        """
        Iteratively select points ensuring minimum distance between them.
        
        Args:
            min_distance: Minimum distance between selected points
            selection_method: 'random' or 'sequential'
        
        Returns:
            filtered_coords, filtered_amplitudes
        """
        if selection_method == 'random':
            # Random order
            sorted_indices = np.random.permutation(self.n_samples)
        else:  # sequential
            sorted_indices = np.arange(self.n_samples)
        
        selected_indices = []
        selected_coords = []
        
        for idx in sorted_indices:
            current_coord = self.coordinates[idx]
            
            # Check if this point is far enough from all selected points
            if len(selected_coords) == 0:
                # First point
                selected_indices.append(idx)
                selected_coords.append(current_coord)
            else:
                # Calculate distances to all selected points
                distances = np.linalg.norm(
                    np.array(selected_coords) - current_coord, axis=1
                )
                
                if np.min(distances) >= min_distance:
                    selected_indices.append(idx)
                    selected_coords.append(current_coord)
        
        selected_indices = np.array(selected_indices)
        return self.coordinates[selected_indices], self.amplitudes[selected_indices]
    
    def analyze_uniformity(self, coords: np.ndarray) -> dict:
        """
        Analyze spatial uniformity of a coordinate set.
        
        Returns:
            Dictionary with uniformity metrics
        """
        if len(coords) < 2:
            return {'mean_distance': 0, 'std_distance': 0, 'cv': 0}
        
        # Calculate pairwise distances
        distances = pdist(coords)
        
        # Find nearest neighbor distances
        tree = cKDTree(coords)
        nn_distances = []
        for i in range(len(coords)):
            dist, _ = tree.query(coords[i], k=2)  # k=2 to exclude self
            nn_distances.append(dist[1])  # Second closest is nearest neighbor
        
        nn_distances = np.array(nn_distances)
        
        return {
            'mean_nn_distance': np.mean(nn_distances),
            'std_nn_distance': np.std(nn_distances),
            'cv_nn_distance': np.std(nn_distances) / np.mean(nn_distances) if np.mean(nn_distances) > 0 else 0,
            'median_nn_distance': np.median(nn_distances),
            'n_samples': len(coords)
        }
