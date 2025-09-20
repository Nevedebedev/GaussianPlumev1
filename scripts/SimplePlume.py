"""
Simplified Gaussian Plume Model without numba dependency
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

def simple_dispersion_coefficients(x_dist: np.ndarray, 
                                 stability_idx: int,
                                 urban: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simplified dispersion coefficients without numba
    """
    # Briggs urban coefficients (simplified to use only one set of parameters)
    c_y = np.array([0.22, 0.16, 0.11, 0.08, 0.06, 0.04])
    p_y = np.array([0.894, 0.894, 0.894, 0.894, 0.894, 0.894])
    c_z = np.array([0.20, 0.12, 0.08, 0.06, 0.03, 0.016])
    p_z = np.array([0.894, 0.894, 0.894, 0.894, 1.0, 1.0])
    
    # Initialize output arrays
    sigma_y = np.zeros_like(x_dist)
    sigma_z = np.zeros_like(x_dist)
    
    # Calculate dispersion for all valid distances
    mask = x_dist > 0
    if np.any(mask):
        valid_x = x_dist[mask]
        sigma_y[mask] = c_y[stability_idx] * valid_x ** p_y[stability_idx]
        sigma_z[mask] = c_z[stability_idx] * valid_x ** p_z[stability_idx]
        
        # Apply mixing height cap for stable conditions
        if stability_idx >= 4:  # E or F stability
            max_sigma_z = 0.8 * 1000  # 80% of typical mixing height
            sigma_z = np.minimum(sigma_z, max_sigma_z)
    
    return sigma_y, sigma_z

def simple_gaussian_plume_3d(
    X, Y, Z,
    Q, U, H,
    xs, ys, zs,
    stability_idx,
    urban=False,
    mixing_height=1000.0,
    deposition_velocity=0.0,
    decay_rate=0.0,
    wind_dir_deg=0.0,
    H_is_effective=True,
    sigma_y0=5.0, 
    sigma_z0=2.0,
    meander_std_deg=0.0
):
    """
    Simplified 3D Gaussian plume model without numba
    """
    # Calculate distances from source
    dx = X - xs
    dy = Y - ys
    dz = Z - zs
    
    # Calculate downwind distance (always positive)
    x_dist = np.abs(dx)
    
    # Get dispersion coefficients
    sigma_y, sigma_z = simple_dispersion_coefficients(x_dist, stability_idx, urban)
    
    # Add initial spread
    sigma_y = np.sqrt(sigma_y**2 + sigma_y0**2)
    sigma_z = np.sqrt(sigma_z**2 + sigma_z0**2)
    
    # Calculate concentration using Gaussian plume equation
    A = Q / (2 * np.pi * U * sigma_y * sigma_z + 1e-10)
    B = np.exp(-0.5 * (dy / sigma_y) ** 2)
    C = np.exp(-0.5 * ((dz - H) / sigma_z) ** 2) + np.exp(-0.5 * ((dz + H) / sigma_z) ** 2)
    
    concentration = A * B * C
    
    # Apply decay
    if decay_rate > 0:
        travel_time = x_dist / (U + 1e-10)
        concentration *= np.exp(-decay_rate * travel_time)
    
    return concentration
