from flask import Flask, request, jsonify, send_from_directory
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import base64
import io
import json

# Add the scripts directory to the path
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(current_dir, 'scripts'))

try:
    from GaussianPlumev2 import vectorized_gaussian_plume_3d, vectorized_dispersion_coefficients
except ImportError as e:
    print(f"Error importing GaussianPlumev2: {e}")
    raise

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

        # Your existing graph generation code here
        # For now, let's just return a test response
        import matplotlib.pyplot as plt
        import io
        import base64
        
        # Create a simple plot
        plt.figure(figsize=(10, 6))
        plt.plot([0, 1, 2, 3, 4], [0, 1, 4, 9, 16])
        plt.title('Test Plot')
        
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
        error_response = jsonify({
            "success": False, 
            "error": str(e)
        })
        error_response.headers.add('Access-Control-Allow-Origin', '*')
        error_response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        error_response.headers.add('Access-Control-Allow-Methods', 'POST')
        return error_response, 400

# This is required for Vercel
handler = app
