from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import tempfile
import base64
from io import BytesIO
from PIL import Image
import cv2
import numpy as np

app = Flask(__name__)
CORS(app)

# Import face matcher if available
try:
    from face_matcher import FaceMatcher
    face_matcher = FaceMatcher()
    FACE_MATCHING_AVAILABLE = True
except ImportError:
    print("Warning: Face matching not available. Install deepface dependencies.")
    FACE_MATCHING_AVAILABLE = False

# Import image detector
try:
    from image_detector import detect_tampering
    TAMPERING_DETECTION_AVAILABLE = True
except ImportError:
    print("Warning: Image tampering detection not available.")
    TAMPERING_DETECTION_AVAILABLE = False


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'face_matching': FACE_MATCHING_AVAILABLE,
        'tampering_detection': TAMPERING_DETECTION_AVAILABLE
    }), 200


@app.route('/detect', methods=['POST'])
def detect_tampering_endpoint():
    """Detect image tampering"""
    if not TAMPERING_DETECTION_AVAILABLE:
        return jsonify({'error': 'Tampering detection not available'}), 503
    
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            file.save(tmp_file.name)
            tmp_path = tmp_file.name
        
        try:
            # Detect tampering
            result = detect_tampering(tmp_path)
            return jsonify(result), 200
        finally:
            # Clean up temp file
            try:
                os.unlink(tmp_path)
            except:
                pass
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/face/detect', methods=['POST'])
def detect_faces():
    """Detect faces in an image"""
    if not FACE_MATCHING_AVAILABLE:
        return jsonify({'error': 'Face detection not available'}), 503
    
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Get optional parameters
        detector = request.form.get('detector', 'retinaface')
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            file.save(tmp_file.name)
            tmp_path = tmp_file.name
        
        try:
            # Detect faces
            result = face_matcher.detect_faces(tmp_path, detector_backend=detector)
            return jsonify(result), 200
        finally:
            # Clean up temp file
            try:
                os.unlink(tmp_path)
            except:
                pass
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/face/search', methods=['POST'])
def search_faces():
    """Search for faces in database"""
    if not FACE_MATCHING_AVAILABLE:
        return jsonify({'error': 'Face matching not available'}), 503
    
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Get optional parameters
        detector = request.form.get('detector', 'retinaface')
        model = request.form.get('model', 'ArcFace')
        threshold = float(request.form.get('threshold', '0.5'))
        database_path = request.form.get('database_path', 'database/')
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            file.save(tmp_file.name)
            tmp_path = tmp_file.name
        
        try:
            # Search for faces
            result = face_matcher.search_faces(
                tmp_path,
                database_path=database_path,
                model_name=model,
                detector_backend=detector,
                threshold=threshold
            )
            return jsonify(result), 200
        finally:
            # Clean up temp file
            try:
                os.unlink(tmp_path)
            except:
                pass
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/face/detect-and-search', methods=['POST'])
def detect_and_search_faces():
    """Detect faces in image and search against database"""
    if not FACE_MATCHING_AVAILABLE:
        return jsonify({'error': 'Face matching not available'}), 503
    
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Get optional parameters
        detector = request.form.get('detector', 'retinaface')
        model = request.form.get('model', 'ArcFace')
        threshold = float(request.form.get('threshold', '0.5'))
        database_path = request.form.get('database_path', 'database/')
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            file.save(tmp_file.name)
            tmp_path = tmp_file.name
        
        try:
            # Detect and search faces
            result = face_matcher.detect_and_search(
                tmp_path,
                database_path=database_path,
                model_name=model,
                detector_backend=detector,
                threshold=threshold
            )
            return jsonify(result), 200
        finally:
            # Clean up temp file
            try:
                os.unlink(tmp_path)
            except:
                pass
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
