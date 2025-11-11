"""
Flask API service for image detection
Deploy this on Railway separately from the frontend
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import tempfile
from image_detector import ImageDetector

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend

# Initialize detector with environment variables
API_USER = os.getenv('IMAGE_DETECTION_API_USER', '1969601374')
API_SECRET = os.getenv('IMAGE_DETECTION_API_SECRET', 'uk7Rwq4Bh8kURjU3WauN3J7nhtGgjSQz')
detector = ImageDetector(API_USER, API_SECRET)

@app.route('/', methods=['GET'])
def root():
    """Root endpoint"""
    return jsonify({
        'status': 'ok',
        'service': 'image-detection-backend',
        'version': '1.0.0',
        'endpoints': {
            'health': '/health',
            'detect': '/detect'
        }
    }), 200

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'service': 'image-detection-backend',
        'version': '1.0.0'
    }), 200

@app.route('/detect', methods=['POST'])
def detect():
    """
    Image detection endpoint
    Accepts multipart/form-data with 'image' field
    """
    try:
        if 'image' not in request.files:
            return jsonify({
                'status': 'error',
                'error': 'No image file provided'
            }), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({
                'status': 'error',
                'error': 'No file selected'
            }), 400
        
        # Validate file type
        if not file.content_type or not file.content_type.startswith('image/'):
            return jsonify({
                'status': 'error',
                'error': 'File must be an image'
            }), 400
        
        # Save temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            file.save(tmp_file.name)
            temp_path = tmp_file.name
        
        try:
            # Analyze image
            results = detector.analyze_image(temp_path)
            
            # Return results
            return jsonify(results), 200
        except Exception as e:
            return jsonify({
                'status': 'error',
                'error': str(e),
                'message': 'Failed to analyze image'
            }), 500
        finally:
            # Clean up temp file
            try:
                os.unlink(temp_path)
            except:
                pass
            
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e),
            'message': 'Internal server error'
        }), 500

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)

