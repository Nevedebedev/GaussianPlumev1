#!/usr/bin/env python3
"""
Flask web application for Gaussian Plume Dispersion Graph Generator
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify, send_from_directory
import base64
import io
import json

# Add the scripts directory to the path to import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))

try:
    from GaussianPlumev2 import vectorized_gaussian_plume_3d, vectorized_dispersion_coefficients
except ImportError:
    print("Error: Could not import GaussianPlumev2. Make sure the file exists in the scripts directory.")
    sys.exit(1)

app = Flask(__name__)

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/generate_graph', methods=['POST'])
def generate_graph():
    try:
        # Get parameters from request
        params = request.json
        
        # Extract and validate parameters
        Q = float(params.get('emission_rate', 100))
        H = float(params.get('stack_height', 50))
        xs = float(params.get('source_x', 50))
        ys = float(params.get('source_y', 0))
        zs = float(params.get('source_z', 10))
        stability_class = params.get('stability_class', 'D')
        U = float(params.get('wind_speed', 5))
        domain_x = float(params.get('domain_x', 2000))
        domain_y = float(params.get('domain_y', 1000))
        wind_dir = float(params.get('wind_direction', 0))
        
        # Validate inputs
        if Q <= 0 or H <= 0 or U <= 0:
            return jsonify({'success': False, 'error': 'Emission rate, stack height, and wind speed must be positive'})
        
        if domain_x < 500 or domain_y < 200:
            return jsonify({'success': False, 'error': 'Domain sizes too small'})
        
        # Map stability class to index
        stability_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5}
        stability_idx = stability_map.get(stability_class, 3)
        
        # Create simulation configuration
        config = {
            'domain': {'x': (0, domain_x), 'y': (-domain_y/2, domain_y/2), 'z': (0, 200)},
            'source_pos': (xs, ys, zs),
            'Q': Q,
            'U': U,
            'H': H,
            'stability': stability_class,
            'urban': False,
            'wind_direction': wind_dir,
            'grid_resolution': (150, 100, 40)  # Reduced for faster computation
        }
        
        # Create grid
        nx, ny, nz = config['grid_resolution']
        x = np.linspace(*config['domain']['x'], nx)
        y = np.linspace(*config['domain']['y'], ny)
        z = np.linspace(*config['domain']['z'], nz)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        # Flatten for calculation
        grid_points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
        
        # Run simulation
        C = vectorized_gaussian_plume_3d(
            X, Y, Z, 
            Q, U, H,
            xs, ys, zs,
            stability_idx,
            wind_dir_deg=wind_dir,
            urban=config['urban'],
            H_is_effective=True
        )
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Ground-level concentration (z = 1.5m)
        slice_z = 1.5
        z_idx = np.argmin(np.abs(z - slice_z))
        C_ground = C[:, :, z_idx]
        
        # Remove zeros for better visualization
        C_ground = np.where(C_ground < 1e-10, np.nan, C_ground)
        
        # Plot 1: Ground-level concentration
        im1 = ax1.contourf(X[:, :, z_idx], Y[:, :, z_idx], C_ground, 
                          levels=20, cmap='viridis', extend='max')
        ax1.set_xlabel('Distance East (m)')
        ax1.set_ylabel('Distance North (m)')
        ax1.set_title(f'Ground-level Concentration (z = {slice_z} m)')
        ax1.plot(xs, ys, 'r*', markersize=15, label='Source')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        plt.colorbar(im1, ax=ax1, label='Concentration (g/m³)')
        
        # Plot 2: Vertical cross-section (y = 0)
        y_idx = np.argmin(np.abs(y))
        C_vertical = C[:, y_idx, :]
        C_vertical = np.where(C_vertical < 1e-10, np.nan, C_vertical)
        
        im2 = ax2.contourf(X[:, y_idx, :], Z[:, y_idx, :], C_vertical, 
                          levels=20, cmap='viridis', extend='max')
        ax2.set_xlabel('Distance East (m)')
        ax2.set_ylabel('Height (m)')
        ax2.set_title(f'Vertical Cross-section (y = {y[y_idx]:.1f} m)')
        ax2.plot(xs, zs, 'r*', markersize=15, label='Source')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        plt.colorbar(im2, ax=ax2, label='Concentration (g/m³)')
        
        plt.tight_layout()
        
        # Convert plot to base64 string
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()
        
        return jsonify({
            'success': True,
            'image': img_base64,
            'max_concentration': float(np.nanmax(C)),
            'parameters_used': {
                'emission_rate': Q,
                'stack_height': H,
                'source_position': [xs, ys, zs],
                'stability_class': stability_class,
                'wind_speed': U,
                'wind_direction': wind_dir,
                'domain_size': [domain_x, domain_y]
            }
        })
        
    except Exception as e:
        import traceback
        error_msg = f"Error generating graph: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    print("Starting Gaussian Plume Dispersion Graph Generator...")
    print("Open your browser to http://localhost:8080")
    app.run(debug=True, host='0.0.0.0', port=8080)
