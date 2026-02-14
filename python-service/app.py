"""
Flask API service for image detection
Deploy this separately and call from Next.js
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
from pathlib import Path
import os

# Add parent directory to path to import image_detector
sys.path.insert(0, str(Path(__file__).parent.parent / 'model'))

from image_detector import ImageDetector

app = Flask(__name__)
CORS(app)  # Enable CORS for Next.js

# Initialize detector
API_USER = os.getenv('IMAGE_DETECTION_API_USER', '1969601374')
API_SECRET = os.getenv('IMAGE_DETECTION_API_SECRET', 'uk7Rwq4Bh8kURjU3WauN3J7nhtGgjSQz')
detector = ImageDetector(API_USER, API_SECRET)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'}), 200

@app.route('/detect', methods=['POST'])
def detect():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Save temporary file
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            file.save(tmp_file.name)
            temp_path = tmp_file.name
        
        try:
            # Analyze image
            results = detector.analyze_image(temp_path)
            return jsonify(results), 200
        finally:
            # Clean up temp file
            os.unlink(temp_path)
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)





