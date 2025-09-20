from flask import Flask, request, jsonify, send_from_directory
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
import io
import base64
import os
import sys
from werkzeug.middleware.proxy_fix import ProxyFix

# Add scripts directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'scripts'))
from SimplePlume import simple_gaussian_plume_3d

app = Flask(__name__, static_folder='../public')

@app.route('/')
def serve():
    return send_from_directory('../public', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('../public', path)

@app.route('/api/generate_graph', methods=['POST', 'OPTIONS'])
def generate_graph():
    if request.method == 'OPTIONS':
        response = jsonify({'message': 'Preflight check passed'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST')
        return response
    try:
        # Get parameters from request
        if not request.is_json:
            raise ValueError("Request must be JSON")
            
        params = request.get_json()
        if not params:
            raise ValueError("No JSON data received")
        
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

        # Your existing graph generation code here
        # Generate concentration field using the simplified function
        X = np.linspace(0, domain_x, 100)
        Y = np.linspace(0, domain_y, 100)
        X, Y = np.meshgrid(X, Y)
        Z = np.zeros_like(X)
        stability_class_idx = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5}.get(stability_class, 3)
        domain_z = 1000.0

        concentration = simple_gaussian_plume_3d(
            X, Y, Z,
            Q, U, H,
            xs, ys, zs,
            stability_class_idx,
            urban=False,
            mixing_height=domain_z,
            deposition_velocity=0.0,
            decay_rate=0.0,
            wind_dir_deg=0.0,
            H_is_effective=True,
            sigma_y0=5.0,
            sigma_z0=2.0,
            meander_std_deg=0.0
        )

        # Create a simple plot
        plt.figure(figsize=(10, 6))
        plt.imshow(concentration, cmap='viridis', origin='lower', extent=[0, domain_x, 0, domain_y])
        plt.title('Concentration Field')
        
        # Save plot to a bytes buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        
        # Convert to base64
        img_str = base64.b64encode(buf.read()).decode('ascii')
        plt.close()
        
        # Create response with CORS headers
        response = jsonify({
            "success": True, 
            "image": img_str
        })
        
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST')
        
        return response
        
    except Exception as e:
        import traceback
        error_details = {
            "success": False,
            "error": str(e),
            "type": type(e).__name__,
            "traceback": traceback.format_exc()
        }
        print(f"Error: {error_details}")  # This will appear in Vercel logs
        
        error_response = jsonify(error_details)
        error_response.headers.add('Access-Control-Allow-Origin', '*')
        error_response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        error_response.headers.add('Access-Control-Allow-Methods', 'POST')
        return error_response, 500

# This is required for Vercel
handler = app
