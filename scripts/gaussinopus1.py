"""
Comprehensive PINN Training Data Generation Optimization
Fully vectorized, parallel, and physics-enhanced implementation for 3D atmospheric benzene dispersion
"""

import numpy as np
try:
    import numba
    from numba import jit, prange, float64, int32
except Exception:
    numba = None
    def jit(*args, **kwargs):
        # Fallback no-op decorator when numba is unavailable
        def wrapper(func):
            return func
        return wrapper
    def prange(*args, **kwargs):
        # Fallback to Python range
        return range(*args, **kwargs)
    # Placeholders for numba dtypes (unused when not jitting)
    float64 = None
    int32 = None
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.spatial import cKDTree
from scipy.stats import qmc
import multiprocessing as mp
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Union
import time
import h5py
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CORE OPTIMIZED DISPERSION FUNCTIONS
# ============================================================================

@jit(nopython=True, parallel=True, cache=True)
def vectorized_dispersion_coefficients(x_prime: np.ndarray, 
                                      stability_idx: int,
                                      urban: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fully vectorized Briggs dispersion coefficients with urban/rural distinction
    
    Parameters:
    -----------
    x_prime : np.ndarray
        Downwind distance from source (m)
    stability_idx : int
        Stability class index (0=A, 1=B, ..., 5=F)
    urban : bool
        Urban (True) or rural (False) environment
    """
    # Briggs urban coefficients
    if urban:
        # Urban dispersion parameters (McElroy-Pooler)
        c_y = np.array([0.32, 0.32, 0.22, 0.16, 0.11, 0.11])
        p_y = np.array([0.78, 0.78, 0.78, 0.78, 0.78, 0.78])
        c_z = np.array([0.24, 0.24, 0.20, 0.14, 0.08, 0.08])
        p_z = np.array([0.71, 0.71, 0.71, 0.71, 0.71, 0.71])
    else:
        # Rural Pasquill-Gifford-Turner with Briggs interpolation
        # Arrays for vectorized computation
        c_y = np.array([0.22, 0.16, 0.11, 0.08, 0.06, 0.04])
        p_y = np.array([0.894, 0.894, 0.894, 0.894, 0.894, 0.894])
        c_z = np.array([0.20, 0.12, 0.08, 0.06, 0.03, 0.016])
        p_z = np.array([0.894, 0.894, 0.894, 0.894, 1.0, 1.0])
    
    # Vectorized calculation
    mask = x_prime > 0
    sigma_y = np.zeros_like(x_prime)
    sigma_z = np.zeros_like(x_prime)
    
    valid_x = x_prime[mask]
    if len(valid_x) > 0:
        # Power law formulation for far field
        sigma_y[mask] = c_y[stability_idx] * valid_x ** p_y[stability_idx]
        sigma_z[mask] = c_z[stability_idx] * valid_x ** p_z[stability_idx]
        
        # Apply mixing height cap for stable conditions
        if stability_idx >= 4:  # E or F stability
            max_sigma_z = 0.8 * 1000  # 80% of typical mixing height
            sigma_z = np.minimum(sigma_z, max_sigma_z)
    
    return sigma_y, sigma_z


@jit(nopython=True, parallel=True, cache=True, fastmath=True)
def vectorized_gaussian_plume_3d(X: np.ndarray, Y: np.ndarray, Z: np.ndarray,
                                Q: float, U: float, H: float,
                                xs: float, ys: float, zs: float,
                                stability_idx: int,
                                urban: bool = False,
                                mixing_height: float = 1000.0,
                                deposition_velocity: float = 0.0,
                                decay_rate: float = 0.0) -> np.ndarray:
    """
    Fully vectorized 3D Gaussian plume model with advanced physics
    
    Parameters:
    -----------
    X, Y, Z : np.ndarray
        3D meshgrid coordinates (m)
    Q : float
        Emission rate (g/s)
    U : float
        Wind speed (m/s)
    H : float
        Effective stack height (m)
    xs, ys, zs : float
        Source coordinates (m)
    stability_idx : int
        Stability class index (0-5)
    urban : bool
        Urban environment flag
    mixing_height : float
        Atmospheric mixing layer height (m)
    deposition_velocity : float
        Ground deposition velocity (m/s)
    decay_rate : float
        First-order decay rate (1/s)
    """
    shape = X.shape
    X_flat = X.ravel()
    Y_flat = Y.ravel()
    Z_flat = Z.ravel()
    
    # Transform to source-relative coordinates
    x_prime = X_flat - xs
    y_prime = Y_flat - ys
    
    # Initialize concentration array
    C = np.zeros(len(X_flat), dtype=np.float64)
    
    # Only calculate for downwind points
    downwind_mask = x_prime > 0
    if not np.any(downwind_mask):
        return C.reshape(shape)
    
    # Extract downwind points
    x_down = x_prime[downwind_mask]
    y_down = y_prime[downwind_mask]
    z_down = Z_flat[downwind_mask]
    
    # Get dispersion coefficients
    sigma_y, sigma_z = vectorized_dispersion_coefficients(x_down, stability_idx, urban)
    
    # Avoid division by zero
    valid_mask = (sigma_y > 0) & (sigma_z > 0)
    if not np.any(valid_mask):
        return C.reshape(shape)
    
    # Extract valid points
    x_valid = x_down[valid_mask]
    y_valid = y_down[valid_mask]
    z_valid = z_down[valid_mask]
    sigma_y_valid = sigma_y[valid_mask]
    sigma_z_valid = sigma_z[valid_mask]
    
    # Calculate concentration components
    # Lateral dispersion
    lateral = np.exp(-0.5 * (y_valid / sigma_y_valid) ** 2)
    
    # Vertical dispersion with ground reflection
    z_minus_H = z_valid - (H + zs)
    z_plus_H = z_valid + (H + zs)
    
    vertical_direct = np.exp(-0.5 * (z_minus_H / sigma_z_valid) ** 2)
    vertical_reflected = np.exp(-0.5 * (z_plus_H / sigma_z_valid) ** 2)
    
    # Add mixing height reflection (multiple reflections)
    vertical_total = vertical_direct + vertical_reflected
    
    if mixing_height > 0 and mixing_height < np.inf:
        # Add reflections from mixing height lid
        for n in range(1, 4):  # Include first 3 reflections
            # Upper reflections
            z_upper = z_valid - (H + zs + 2 * n * mixing_height)
            vertical_total += np.exp(-0.5 * (z_upper / sigma_z_valid) ** 2)
            
            # Lower reflections  
            z_lower = z_valid + (H + zs - 2 * n * mixing_height)
            vertical_total += np.exp(-0.5 * (z_lower / sigma_z_valid) ** 2)
    
    # Main concentration calculation
    prefactor = Q / (2 * np.pi * U * sigma_y_valid * sigma_z_valid)
    C_valid = prefactor * lateral * vertical_total
    
    # Apply decay if specified
    if decay_rate > 0:
        travel_time = x_valid / U
        C_valid *= np.exp(-decay_rate * travel_time)
    
    # Apply deposition if specified
    if deposition_velocity > 0 and np.any(z_valid < 10):  # Near ground
        deposition_factor = np.exp(-deposition_velocity * x_valid / (U * H))
        C_valid *= deposition_factor
    
    # Place valid concentrations back
    valid_indices = np.where(downwind_mask)[0][valid_mask]
    for i in prange(len(valid_indices)):
        C[valid_indices[i]] = C_valid[i]
    
    return C.reshape(shape)


# ============================================================================
# ADVANCED PHYSICS MODULES
# ============================================================================

@jit(nopython=True, cache=True)
def calculate_plume_rise(Q: float, U: float, Ts: float, Ta: float, 
                        stack_diameter: float, exit_velocity: float,
                        stability_idx: int) -> float:
    """
    Calculate plume rise using Briggs formulas
    
    Parameters:
    -----------
    Q : float
        Heat emission rate (MW)
    U : float
        Wind speed (m/s)
    Ts : float
        Stack temperature (K)
    Ta : float
        Ambient temperature (K)
    stack_diameter : float
        Stack diameter (m)
    exit_velocity : float
        Stack exit velocity (m/s)
    stability_idx : int
        Stability class index
    """
    g = 9.81  # gravity (m/s²)
    
    # Buoyancy flux
    F = g * exit_velocity * stack_diameter**2 * (Ts - Ta) / (4 * Ts)
    
    # Momentum flux
    M = exit_velocity * stack_diameter**2 / 4
    
    if stability_idx <= 3:  # Unstable/neutral (A-D)
        # Briggs formula for unstable/neutral conditions
        if F > 0 and F < 55:
            delta_h = 21.425 * (F**0.75) / U
        elif F >= 55:
            delta_h = 38.71 * (F**0.6) / U
        else:
            # Momentum rise only
            delta_h = 3 * (M / U)**0.333
    else:  # Stable (E-F)
        # Calculate stability parameter
        s = g * 0.01 / Ta  # Approximate stability parameter
        
        # Briggs formula for stable conditions
        if F > 0:
            delta_h = 2.6 * (F / (U * s))**(1/3)
        else:
            delta_h = 1.5 * (M / U)**0.333
    
    return delta_h


@jit(nopython=True, parallel=True, cache=True)
def building_wake_effects(X: np.ndarray, Y: np.ndarray, Z: np.ndarray,
                         building_height: float, building_width: float,
                         building_x: float, building_y: float) -> np.ndarray:
    """
    Calculate building wake effects on dispersion
    
    Returns correction factor for concentration
    """
    shape = X.shape
    X_flat = X.ravel()
    Y_flat = Y.ravel()
    Z_flat = Z.ravel()
    
    wake_factor = np.ones(len(X_flat), dtype=np.float64)
    
    # Define wake region (simplified Huber-Snyder approach)
    wake_length = 10 * building_height
    wake_width = 2 * building_width
    
    for i in prange(len(X_flat)):
        x_rel = X_flat[i] - building_x
        y_rel = Y_flat[i] - building_y
        
        # Check if point is in wake region
        if (0 < x_rel < wake_length and 
            abs(y_rel) < wake_width/2 and 
            Z_flat[i] < 2*building_height):
            
            # Enhanced turbulence in wake
            distance_factor = 1 - x_rel / wake_length
            height_factor = 1 - Z_flat[i] / (2 * building_height)
            
            # Increase dispersion by factor of 2-5 in wake
            wake_factor[i] = 1 + 4 * distance_factor * height_factor
    
    return wake_factor.reshape(shape)


# ============================================================================
# ADAPTIVE GRID GENERATION
# ============================================================================

class AdaptiveGrid:
    """
    Generate adaptive training grid with refinement near sources and boundaries
    """
    
    def __init__(self, domain_bounds: Dict[str, Tuple[float, float]],
                 source_locations: List[Tuple[float, float, float]],
                 refinement_levels: Dict[str, int] = None):
        """
        Parameters:
        -----------
        domain_bounds : dict
            {'x': (xmin, xmax), 'y': (ymin, ymax), 'z': (zmin, zmax)}
        source_locations : list
            List of (x, y, z) source coordinates
        refinement_levels : dict
            {'near': points, 'mid': points, 'far': points}
        """
        self.bounds = domain_bounds
        self.sources = np.array(source_locations)
        self.refinement = refinement_levels or {
            'near': 1000,   # Within 100m of sources
            'mid': 2000,    # 100m - 1km
            'far': 1000     # > 1km
        }
    
    def generate_adaptive_points(self, total_points: int = 10000) -> np.ndarray:
        """
        Generate adaptive point distribution using octree-like refinement
        """
        points_list = []
        
        # Near-field points (high density near sources)
        near_points = self._generate_near_field(self.refinement['near'])
        points_list.append(near_points)
        
        # Mid-field points (moderate density)
        mid_points = self._generate_mid_field(self.refinement['mid'])
        points_list.append(mid_points)
        
        # Far-field points (low density)
        far_points = self._generate_far_field(self.refinement['far'])
        points_list.append(far_points)
        
        # Boundary points (ensure boundary conditions are captured)
        boundary_points = self._generate_boundary_points(500)
        points_list.append(boundary_points)
        
        # Combine all points
        all_points = np.vstack(points_list)
        
        # Remove duplicates using KDTree
        tree = cKDTree(all_points)
        unique_mask = np.array([True] * len(all_points))
        
        for i in range(len(all_points)):
            if unique_mask[i]:
                neighbors = tree.query_ball_point(all_points[i], r=1e-6)
                for j in neighbors[1:]:  # Skip self
                    unique_mask[j] = False
        
        return all_points[unique_mask]
    
    def _generate_near_field(self, n_points: int) -> np.ndarray:
        """Generate high-density points near sources"""
        points = []
        points_per_source = n_points // len(self.sources)
        
        for source in self.sources:
            # Exponential spacing in radial direction
            r = np.random.exponential(scale=50, size=points_per_source)
            r = np.clip(r, 1, 200)  # 1-200m from source
            
            # Random angles
            theta = np.random.uniform(0, 2*np.pi, points_per_source)
            phi = np.random.uniform(0, np.pi, points_per_source)
            
            # Convert to Cartesian
            x = source[0] + r * np.sin(phi) * np.cos(theta)
            y = source[1] + r * np.sin(phi) * np.sin(theta)
            z = source[2] + r * np.cos(phi)
            
            # Clip to domain
            x = np.clip(x, self.bounds['x'][0], self.bounds['x'][1])
            y = np.clip(y, self.bounds['y'][0], self.bounds['y'][1])
            z = np.clip(z, self.bounds['z'][0], self.bounds['z'][1])
            
            points.append(np.column_stack([x, y, z]))
        
        return np.vstack(points) if points else np.empty((0, 3))
    
    def _generate_mid_field(self, n_points: int) -> np.ndarray:
        """Generate moderate-density points in mid-field"""
        # Use Latin Hypercube Sampling for better coverage
        sampler = qmc.LatinHypercube(d=3)
        sample = sampler.random(n=n_points)
        
        # Scale to domain
        x = sample[:, 0] * (self.bounds['x'][1] - self.bounds['x'][0]) + self.bounds['x'][0]
        y = sample[:, 1] * (self.bounds['y'][1] - self.bounds['y'][0]) + self.bounds['y'][0]
        z = sample[:, 2] * (self.bounds['z'][1] - self.bounds['z'][0]) + self.bounds['z'][0]
        
        points = np.column_stack([x, y, z])
        
        # Filter to mid-field region (100m - 1km from any source)
        mask = np.zeros(len(points), dtype=bool)
        for i, point in enumerate(points):
            distances = np.linalg.norm(self.sources - point, axis=1)
            if np.any((distances > 100) & (distances < 1000)):
                mask[i] = True
        
        return points[mask]
    
    def _generate_far_field(self, n_points: int) -> np.ndarray:
        """Generate low-density points in far-field"""
        # Sobol sequence for quasi-random sampling
        sampler = qmc.Sobol(d=3, scramble=True)
        sample = sampler.random(n=n_points)
        
        # Scale to domain
        x = sample[:, 0] * (self.bounds['x'][1] - self.bounds['x'][0]) + self.bounds['x'][0]
        y = sample[:, 1] * (self.bounds['y'][1] - self.bounds['y'][0]) + self.bounds['y'][0]
        z = sample[:, 2] * (self.bounds['z'][1] - self.bounds['z'][0]) + self.bounds['z'][0]
        
        points = np.column_stack([x, y, z])
        
        # Filter to far-field region (>1km from all sources)
        mask = np.zeros(len(points), dtype=bool)
        for i, point in enumerate(points):
            distances = np.linalg.norm(self.sources - point, axis=1)
            if np.all(distances > 1000):
                mask[i] = True
        
        return points[mask]
    
    def _generate_boundary_points(self, n_points: int) -> np.ndarray:
        """Generate points on domain boundaries"""
        points = []
        points_per_face = n_points // 6
        
        # Generate points on each face
        for dim in ['x', 'y', 'z']:
            for bound_idx in [0, 1]:
                face_points = self._sample_boundary_face(dim, bound_idx, points_per_face)
                points.append(face_points)
        
        return np.vstack(points) if points else np.empty((0, 3))
    
    def _sample_boundary_face(self, dim: str, bound_idx: int, n_points: int) -> np.ndarray:
        """Sample points on a boundary face"""
        points = np.zeros((n_points, 3))
        
        if dim == 'x':
            points[:, 0] = self.bounds['x'][bound_idx]
            points[:, 1] = np.random.uniform(self.bounds['y'][0], self.bounds['y'][1], n_points)
            points[:, 2] = np.random.uniform(self.bounds['z'][0], self.bounds['z'][1], n_points)
        elif dim == 'y':
            points[:, 0] = np.random.uniform(self.bounds['x'][0], self.bounds['x'][1], n_points)
            points[:, 1] = self.bounds['y'][bound_idx]
            points[:, 2] = np.random.uniform(self.bounds['z'][0], self.bounds['z'][1], n_points)
        else:  # z
            points[:, 0] = np.random.uniform(self.bounds['x'][0], self.bounds['x'][1], n_points)
            points[:, 1] = np.random.uniform(self.bounds['y'][0], self.bounds['y'][1], n_points)
            points[:, 2] = self.bounds['z'][bound_idx]
        
        return points


# ============================================================================
# PINN-SPECIFIC DATA STRUCTURES
# ============================================================================

@dataclass
class PINNTrainingData:
    """
    Organized data structure for PINN training with automatic normalization
    """
    # Spatial coordinates
    coordinates: np.ndarray  # [N, 3] - (x, y, z)
    
    # Physical parameters
    parameters: np.ndarray   # [N, M] - (Q, U, H, stability, ...)
    
    # Target concentrations
    concentrations: np.ndarray  # [N, 1]
    
    # Physics residual points
    residual_points: Optional[np.ndarray] = None
    
    # Boundary condition points
    boundary_points: Optional[np.ndarray] = None
    boundary_values: Optional[np.ndarray] = None
    
    # Data normalization parameters
    coord_mean: Optional[np.ndarray] = None
    coord_std: Optional[np.ndarray] = None
    param_mean: Optional[np.ndarray] = None
    param_std: Optional[np.ndarray] = None
    conc_mean: Optional[float] = None
    conc_std: Optional[float] = None
    
    def normalize(self):
        """Normalize all data for neural network training"""
        # Coordinates normalization
        self.coord_mean = np.mean(self.coordinates, axis=0)
        self.coord_std = np.std(self.coordinates, axis=0) + 1e-8
        self.coordinates = (self.coordinates - self.coord_mean) / self.coord_std
        
        # Parameters normalization
        self.param_mean = np.mean(self.parameters, axis=0)
        self.param_std = np.std(self.parameters, axis=0) + 1e-8
        self.parameters = (self.parameters - self.param_mean) / self.param_std
        
        # Log-transform concentrations (common for dispersion)
        log_conc = np.log10(self.concentrations + 1e-10)
        self.conc_mean = np.mean(log_conc)
        self.conc_std = np.std(log_conc) + 1e-8
        self.concentrations = (log_conc - self.conc_mean) / self.conc_std
        
        # Normalize residual and boundary points if present
        if self.residual_points is not None:
            self.residual_points = (self.residual_points - self.coord_mean) / self.coord_std
        
        if self.boundary_points is not None:
            self.boundary_points = (self.boundary_points - self.coord_mean) / self.coord_std
    
    def denormalize_concentration(self, normalized_conc: np.ndarray) -> np.ndarray:
        """Convert normalized concentration back to physical units"""
        log_conc = normalized_conc * self.conc_std + self.conc_mean
        return 10**log_conc - 1e-10
    
    def create_physics_loss_points(self, n_residual: int = 5000):
        """
        Generate collocation points for physics-informed loss
        """
        # Use adaptive sampling for residual points
        bounds = {
            'x': (np.min(self.coordinates[:, 0]), np.max(self.coordinates[:, 0])),
            'y': (np.min(self.coordinates[:, 1]), np.max(self.coordinates[:, 1])),
            'z': (np.min(self.coordinates[:, 2]), np.max(self.coordinates[:, 2]))
        }
        
        # Generate residual points with higher density near sources
        sampler = qmc.Sobol(d=3, scramble=True)
        self.residual_points = sampler.random(n=n_residual)
        
        # Scale to domain
        for i, dim in enumerate(['x', 'y', 'z']):
            self.residual_points[:, i] = (self.residual_points[:, i] * 
                                         (bounds[dim][1] - bounds[dim][0]) + 
                                         bounds[dim][0])
    
    def get_training_batch(self, batch_size: int, 
                          data_weight: float = 0.5,
                          physics_weight: float = 0.3,
                          boundary_weight: float = 0.2):
        """
        Get balanced batch for PINN training
        """
        n_data = int(batch_size * data_weight)
        n_physics = int(batch_size * physics_weight)
        n_boundary = int(batch_size * boundary_weight)
        
        # Random sampling from each category
        data_idx = np.random.choice(len(self.coordinates), n_data, replace=False)
        
        batch = {
            'data_coords': self.coordinates[data_idx],
            'data_params': self.parameters[data_idx],
            'data_conc': self.concentrations[data_idx]
        }
        
        if self.residual_points is not None and n_physics > 0:
            physics_idx = np.random.choice(len(self.residual_points), n_physics, replace=False)
            batch['physics_points'] = self.residual_points[physics_idx]
        
        if self.boundary_points is not None and n_boundary > 0:
            boundary_idx = np.random.choice(len(self.boundary_points), n_boundary, replace=False)
            batch['boundary_points'] = self.boundary_points[boundary_idx]
            batch['boundary_values'] = self.boundary_values[boundary_idx]
        
        return batch
    
    def save_to_hdf5(self, filename: str):
        """Save training data to HDF5 for efficient loading"""
        with h5py.File(filename, 'w') as f:
            # Save main data
            f.create_dataset('coordinates', data=self.coordinates, compression='gzip')
            f.create_dataset('parameters', data=self.parameters, compression='gzip')
            f.create_dataset('concentrations', data=self.concentrations, compression='gzip')
            
            # Save normalization parameters
            if self.coord_mean is not None:
                norm_group = f.create_group('normalization')
                norm_group.create_dataset('coord_mean', data=self.coord_mean)
                norm_group.create_dataset('coord_std', data=self.coord_std)
                norm_group.create_dataset('param_mean', data=self.param_mean)
                norm_group.create_dataset('param_std', data=self.param_std)
                norm_group.attrs['conc_mean'] = self.conc_mean
                norm_group.attrs['conc_std'] = self.conc_std
            
            # Save physics points if present
            if self.residual_points is not None:
                f.create_dataset('residual_points', data=self.residual_points, compression='gzip')
            
            if self.boundary_points is not None:
                f.create_dataset('boundary_points', data=self.boundary_points, compression='gzip')
                f.create_dataset('boundary_values', data=self.boundary_values, compression='gzip')
    
    @classmethod
    def load_from_hdf5(cls, filename: str):
        """Load training data from HDF5"""
        with h5py.File(filename, 'r') as f:
            data = cls(
                coordinates=f['coordinates'][:],
                parameters=f['parameters'][:],
                concentrations=f['concentrations'][:]
            )
            
            # Load normalization parameters if present
            if 'normalization' in f:
                norm = f['normalization']
                data.coord_mean = norm['coord_mean'][:]
                data.coord_std = norm['coord_std'][:]
                data.param_mean = norm['param_mean'][:]
                data.param_std = norm['param_std'][:]
                data.conc_mean = norm.attrs['conc_mean']
                data.conc_std = norm.attrs['conc_std']
            
            # Load physics points if present
            if 'residual_points' in f:
                data.residual_points = f['residual_points'][:]
            
            if 'boundary_points' in f:
                data.boundary_points = f['boundary_points'][:]
                data.boundary_values = f['boundary_values'][:]
            
            return data


# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_concentration_slice(grid_points: np.ndarray, 
                                 concentrations: np.ndarray,
                                 slice_dim: str = 'z', 
                                 slice_val: float = 1.5,
                                 source_pos: Optional[Tuple[float, float, float]] = None):
    """
    Visualize a 2D slice of the concentration field as a heatmap.
    
    Parameters:
    -----------
    grid_points : np.ndarray
        [N, 3] array of (x, y, z) coordinates.
    concentrations : np.ndarray
        [N, 1] array of concentration values.
    slice_dim : str
        Dimension to slice ('x', 'y', or 'z').
    slice_val : float
        Value at which to slice the dimension.
    source_pos : tuple, optional
        (xs, ys, zs) coordinates of the source to plot.
    """
    # Determine the structured grid dimensions from unique coordinates
    x_vals = np.unique(grid_points[:, 0])
    y_vals = np.unique(grid_points[:, 1])
    z_vals = np.unique(grid_points[:, 2])
    nx, ny, nz = len(x_vals), len(y_vals), len(z_vals)

    # Reshape concentrations back to (nx, ny, nz)
    c_full = concentrations.ravel().reshape(nx, ny, nz)

    # Find the closest grid value to the requested slice
    if slice_dim == 'z':
        dim_idx = 2
        xlabel, ylabel = 'Downwind Distance (m)', 'Crosswind Distance (m)'
        k = np.abs(z_vals - slice_val).argmin()
        closest_val = z_vals[k]
        Zslice = c_full[:, :, k]
        XX, YY = np.meshgrid(x_vals, y_vals, indexing='ij')
        Xp, Yp, Cp = XX, YY, Zslice
    elif slice_dim == 'y':
        dim_idx = 1
        xlabel, ylabel = 'Downwind Distance (m)', 'Height (m)'
        j = np.abs(y_vals - slice_val).argmin()
        closest_val = y_vals[j]
        Yslice = c_full[:, j, :]
        XX, ZZ = np.meshgrid(x_vals, z_vals, indexing='ij')
        Xp, Yp, Cp = XX, ZZ, Yslice
    else:  # slice_dim == 'x'
        dim_idx = 0
        xlabel, ylabel = 'Crosswind Distance (m)', 'Height (m)'
        i = np.abs(x_vals - slice_val).argmin()
        closest_val = x_vals[i]
        Xslice = c_full[i, :, :]
        YY, ZZ = np.meshgrid(y_vals, z_vals, indexing='ij')
        Xp, Yp, Cp = YY, ZZ, Xslice

    # Plotting directly from structured grid
    plt.figure(figsize=(12, 8))
    plt.contourf(Xp, Yp, Cp, levels=15, cmap='viridis')
    plt.colorbar(label='Concentration (g/m³)')
    
    # Plot source position
    if source_pos:
        if slice_dim == 'z':
            plt.plot(source_pos[0], source_pos[1], 'r*', markersize=15, label='Source')
        elif slice_dim == 'y':
            plt.plot(source_pos[0], source_pos[2], 'r*', markersize=15, label='Source')
        elif slice_dim == 'x':
            plt.plot(source_pos[1], source_pos[2], 'r*', markersize=15, label='Source')
        plt.legend()

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f'Concentration Slice at {slice_dim.upper()} = {closest_val:.2f} m')
    plt.grid(True, linestyle='--', alpha=0.6)
    # Save first for headless environments
    out_name = f'heatmap_{slice_dim}.png'
    try:
        plt.savefig(out_name, dpi=200, bbox_inches='tight')
        print(f"Saved heatmap to {out_name}")
    except Exception as e:
        print(f"Warning: could not save figure: {e}")
    plt.show()


# ============================================================================
# PARALLEL SCENARIO GENERATION
# ============================================================================

def process_single_scenario(args):
    """
    Process a single dispersion scenario (for parallel processing)
    """
    scenario, grid_points, config = args
    
    # Extract parameters
    Q = scenario['Q']
    U = scenario['U']
    H = scenario['H']
    stability_idx = scenario['stability_idx']
    xs, ys, zs = scenario['source_pos']
    
    # Reshape grid points for vectorized calculation
    n_points = len(grid_points)
    X = grid_points[:, 0].reshape(n_points, 1, 1)
    Y = grid_points[:, 1].reshape(n_points, 1, 1)
    Z = grid_points[:, 2].reshape(n_points, 1, 1)
    
    # Calculate concentrations
    C = vectorized_gaussian_plume_3d(
        X, Y, Z, Q, U, H, xs, ys, zs,
        stability_idx,
        urban=config.get('urban', False),
        mixing_height=config.get('mixing_height', 1000),
        deposition_velocity=config.get('deposition_velocity', 0),
        decay_rate=config.get('decay_rate', 0)
    )
    
    return C.flatten()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == '__main__':
    # --- 1. Define Simulation Configuration ---
    config = {
        'domain': {'x': (0, 2000), 'y': (-500, 500), 'z': (0, 200)},
        'source_pos': (50, 0, 10), # (x, y, z)
        'Q': 100.0,  # Emission rate (g/s)
        'U': 5.0,    # Wind speed (m/s)
        'H': 50.0,   # Effective stack height (m)
        'stability': 'D', # Pasquill stability class (A-F)
        'urban': False,
        'grid_resolution': (200, 100, 50) # (nx, ny, nz)
    }

    stability_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5}
    stability_idx = stability_map[config['stability']]

    # --- 2. Create a Regular Grid for Visualization ---
    nx, ny, nz = config['grid_resolution']
    x = np.linspace(*config['domain']['x'], nx)
    y = np.linspace(*config['domain']['y'], ny)
    z = np.linspace(*config['domain']['z'], nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    grid_points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T

    # --- 3. Define a Single Scenario ---
    scenario = {
        'Q': config['Q'],
        'U': config['U'],
        'H': config['H'],
        'stability_idx': stability_idx,
        'source_pos': config['source_pos']
    }

    # --- 4. Run the Simulation ---
    print("Running Gaussian plume simulation...")
    start_time = time.time()
    
    # Reshape for the vectorized function
    C = vectorized_gaussian_plume_3d(
        X, Y, Z, 
        scenario['Q'], scenario['U'], scenario['H'],
        scenario['source_pos'][0], scenario['source_pos'][1], scenario['source_pos'][2],
        scenario['stability_idx'],
        urban=config['urban']
    )
    
    end_time = time.time()
    print(f"Simulation finished in {end_time - start_time:.2f} seconds.")

    # --- 5. Visualize the Results ---
    print("Generating heatmap...")
    # Ground-level concentration slice
    visualize_concentration_slice(grid_points, C.flatten(), 
                                  slice_dim='z', 
                                  slice_val=1.5, 
                                  source_pos=config['source_pos'])

    # Centerline vertical slice
    visualize_concentration_slice(grid_points, C.flatten(), 
                                  slice_dim='y', 
                                  slice_val=0, 
                                  source_pos=config['source_pos'])