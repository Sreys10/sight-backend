from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import tempfile
import base64
import subprocess
import json
from io import BytesIO
from PIL import Image
import cv2
import numpy as np
from functools import wraps

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, use system environment variables

app = Flask(__name__)
CORS(app)

# Security: Require API key from callers
API_KEY = os.getenv('EVI_CHECK_API_KEY', 'default-api-key-replace-me')

def require_api_key(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        key = request.headers.get('X-API-Key')
        if key and key == API_KEY:
            return f(*args, **kwargs)
        return jsonify({'error': 'Unauthorized: Invalid or missing API Key'}), 401
    return decorated

# Import face matcher if available
try:
    from face_matcher import FaceMatcher
    face_matcher = FaceMatcher()
    FACE_MATCHING_AVAILABLE = True
except Exception as e:
    import traceback
    with open('face_matcher_error.log', 'w') as f:
        f.write(f"Exception raised during FaceMatcher load: {str(e)}\n")
        traceback.print_exc(file=f)
    print(f"Warning: Face matching not available. Error: {str(e)}")
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
    faiss_status = False
    if FACE_MATCHING_AVAILABLE:
        stats = face_matcher.get_index_stats()
        faiss_status = stats['indexed']
    
    return jsonify({
        'status': 'healthy',
        'face_matching': FACE_MATCHING_AVAILABLE,
        'faiss_indexed': faiss_status,
        'tampering_detection': TAMPERING_DETECTION_AVAILABLE
    }), 200


@app.route('/detect', methods=['POST'])
@require_api_key
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
@require_api_key
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
@require_api_key
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
            # Get optional parameter for FAISS usage
            use_faiss = request.form.get('use_faiss', 'true').lower() == 'true'
            
            # Search for faces
            result = face_matcher.search_faces(
                tmp_path,
                database_path=database_path,
                model_name=model,
                detector_backend=detector,
                threshold=threshold,
                use_faiss=use_faiss
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
@require_api_key
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
        use_faiss = request.form.get('use_faiss', 'true').lower() == 'true'
        
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
                threshold=threshold,
                use_faiss=use_faiss
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


@app.route('/face/index-database', methods=['POST'])
@require_api_key
def index_database():
    """Index all images in the database folder to FAISS"""
    if not FACE_MATCHING_AVAILABLE:
        return jsonify({'error': 'Face matching not available'}), 503
    
    try:
        # Get optional parameters
        database_path = request.form.get('database_path', 'database/')
        model = request.form.get('model', 'ArcFace')
        detector = request.form.get('detector', 'retinaface')
        
        # Index database images
        result = face_matcher.index_database_images(
            database_path=database_path,
            model_name=model,
            detector_backend=detector
        )
        
        return jsonify(result), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


db_manager = None
DATABASE_MANAGER_AVAILABLE = False

def get_database_manager():
    global db_manager, DATABASE_MANAGER_AVAILABLE
    try:
        from database_manager import DatabaseManager
        database_url = os.getenv('DATABASE_URL')
        if not database_url:
            return None, "DATABASE_URL environment variable is not set."
        
        if db_manager is None:
            db_manager = DatabaseManager(database_url=database_url)
            
        if db_manager.cursor is None:
            # Connection might have dropped, try to re-initialize
            db_manager = DatabaseManager(database_url=database_url)
            
        if db_manager.cursor is None:
            return None, "PostgreSQL connection failed (cursor is None). Please check database status and connection string."
            
        DATABASE_MANAGER_AVAILABLE = True
        return db_manager, None
    except ImportError as e:
        import traceback
        return None, f"ImportError: Failed to import database_manager dependency. Check that psycopg2-binary is installed. Detail: {str(e)}\n{traceback.format_exc()}"
    except Exception as e:
        import traceback
        return None, f"Initialization failed: {str(e)}\n{traceback.format_exc()}"


@app.route('/face/database/list', methods=['GET'])
@require_api_key
def list_database():
    """List all persons in the database.
    
    Returns metadata + a compact 80x80 thumbnail instead of full base64 images
    to minimise bandwidth through the Vercel proxy layer.  The full image_base64
    is still available via /face/database/person/<id>.
    """
    manager, err = get_database_manager()
    if err:
        return jsonify({'error': f'Database manager not available: {err}'}), 503

    def _make_thumbnail(b64_str: str, size: int = 80) -> str:
        """Downscale a base64 image to a tiny thumbnail (size x size JPEG)."""
        try:
            raw = b64_str.split(',', 1)[-1]   # strip data-URI header if present
            img_data = base64.b64decode(raw)
            nparr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                return b64_str
            h, w = img.shape[:2]
            scale = size / max(h, w)
            thumb = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))),
                               interpolation=cv2.INTER_AREA)
            _, buf = cv2.imencode('.jpg', thumb, [cv2.IMWRITE_JPEG_QUALITY, 40])
            thumb_b64 = base64.b64encode(buf).decode('utf-8')
            return f'data:image/jpeg;base64,{thumb_b64}'
        except Exception:
            return b64_str   # fall back to original on error

    try:
        persons = manager.get_all_persons()
        compact = []
        for p in persons:
            entry = {k: v for k, v in p.items() if k != 'image_base64'}
            # Prefer a Cloudinary URL (no base64 data over the wire at all)
            if p.get('image_url'):
                entry['image_thumbnail'] = p['image_url']
                entry['has_image'] = True
            elif p.get('image_base64'):
                entry['image_thumbnail'] = _make_thumbnail(p['image_base64'])
                entry['has_image'] = True
            else:
                entry['image_thumbnail'] = None
                entry['has_image'] = False
            compact.append(entry)
        return jsonify({
            'success': True,
            'persons': compact
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/face/database/person/<person_id>', methods=['GET'])
@require_api_key
def get_person_detail(person_id):
    """Get full person record including image_base64 for a single person."""
    manager, err = get_database_manager()
    if err:
        return jsonify({'error': f'Database manager not available: {err}'}), 503
    try:
        person = manager.get_person(person_id)
        if not person:
            return jsonify({'error': 'Person not found'}), 404
        return jsonify({'success': True, 'person': person}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/face/database/add', methods=['POST'])
@require_api_key
def add_to_database():
    """Add a person to the database"""
    manager, err = get_database_manager()
    if err:
        return jsonify({'error': f'Database manager not available: {err}'}), 503
    
    try:
        # Debug: Print what Flask received
        print(f"Request method: {request.method}")
        print(f"Content-Type: {request.content_type}")
        print(f"Files keys: {list(request.files.keys())}")
        print(f"Form keys: {list(request.form.keys())}")
        
        # Check if file is in files
        if 'image' not in request.files:
            print("ERROR: 'image' not in request.files")
            print(f"Available files: {list(request.files.keys())}")
            return jsonify({'error': 'No image file provided', 'debug': {'files': list(request.files.keys()), 'form': list(request.form.keys())}}), 400
        
        file = request.files['image']
        if file.filename == '' or not file.filename:
            print("ERROR: File filename is empty")
            return jsonify({'error': 'No file selected'}), 400
        
        person_name = request.form.get('person_name', '').strip()
        if not person_name:
            print("ERROR: Person name is required")
            return jsonify({'error': 'Person name is required'}), 400
        
        print(f"Adding person: {person_name}, File: {file.filename}, Size: {file.content_length if hasattr(file, 'content_length') else 'unknown'}")
        
        result = manager.add_person(
            image_file=file,
            person_name=person_name,
            name=request.form.get('name', ''),
            age=request.form.get('age'),
            email=request.form.get('email', ''),
            phone=request.form.get('phone', ''),
            notes=request.form.get('notes', ''),
            added_by_name=request.form.get('added_by_name', ''),
            added_by_email=request.form.get('added_by_email', ''),
            image_url=request.form.get('image_url', None) or None   # Cloudinary URL (optional)
        )
        
        print(f"Add person result: {result}")
        
        if result.get('success'):
            # Add only the new image to FAISS index (incremental indexing)
            if FACE_MATCHING_AVAILABLE:
                try:
                    # Get the local image path where the file was saved
                    person_name = request.form.get('person_name', '').strip()
                    # Get image extension from the result or determine from file
                    image_ext = result.get('person', {}).get('image_extension', '.jpg')
                    if not image_ext or image_ext == '':
                        # Try to get from file
                        if hasattr(file, 'filename') and file.filename:
                            image_ext = os.path.splitext(file.filename)[1] or '.jpg'
                        else:
                            image_ext = '.jpg'
                    local_path = os.path.join(manager.database_path, f"{person_name}{image_ext}")
                    
                    # Prepare additional metadata
                    additional_metadata = {
                        'name': request.form.get('name', ''),
                        'age': request.form.get('age'),
                        'email': request.form.get('email', ''),
                        'phone': request.form.get('phone', ''),
                        'notes': request.form.get('notes', ''),
                        'added_by': {
                            'name': request.form.get('added_by_name', ''),
                            'email': request.form.get('added_by_email', '')
                        } if request.form.get('added_by_name') else None,
                        'added_at': result.get('person', {}).get('added_at')
                    }
                    
                    # Add single image to index
                    index_result = face_matcher.add_single_image_to_index(
                        image_path=local_path,
                        person_name=person_name,
                        additional_metadata=additional_metadata
                    )
                    
                    if index_result.get('success'):
                        print(f"✓ Added {person_name} to FAISS index (incremental)")
                    else:
                        print(f"Warning: Failed to add to FAISS index: {index_result.get('error')}")
                except Exception as e:
                    print(f"Warning: Failed to add image to FAISS index: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    # Indexing can fail, but person is still added
        
        if not result.get('success'):
            return jsonify(result), 400
        
        return jsonify(result), 200
    except Exception as e:
        import traceback
        print(f"Error in add_to_database: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/face/database/update/<person_id>', methods=['PUT'])
@require_api_key
def update_person(person_id):
    """Update person metadata"""
    manager, err = get_database_manager()
    if err:
        return jsonify({'error': f'Database manager not available: {err}'}), 503
    
    try:
        data = request.json
        result = manager.update_person(
            person_id=person_id,
            person_name=data.get('person_name', ''),
            name=data.get('name', ''),
            age=data.get('age'),
            email=data.get('email', ''),
            phone=data.get('phone', ''),
            notes=data.get('notes', '')
        )
        
        return jsonify(result), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/face/database/delete/<person_id>', methods=['DELETE'])
@require_api_key
def delete_person(person_id):
    """Delete a person from the database"""
    manager, err = get_database_manager()
    if err:
        return jsonify({'error': f'Database manager not available: {err}'}), 503
    
    try:
        # Get person_name before deletion (needed for FAISS index removal)
        person_name = None
        if FACE_MATCHING_AVAILABLE:
            try:
                person = manager.get_person(person_id)
                if person:
                    person_name = person.get('person_name')
            except:
                pass
        
        # If not found, try to extract from person_id
        if not person_name:
            # person_id format is usually: person_name_timestamp
            if '_' in person_id:
                # Extract person_name (everything before last underscore and timestamp)
                parts = person_id.rsplit('_', 1)
                if len(parts) == 2 and parts[1].isdigit():
                    person_name = parts[0]
                else:
                    person_name = person_id
            else:
                person_name = person_id
        
        result = manager.delete_person(person_id)
        
        if result['success']:
            # Remove from FAISS index (incremental removal)
            if FACE_MATCHING_AVAILABLE and person_name:
                try:
                    remove_result = face_matcher.remove_from_index(person_name)
                    if remove_result.get('success'):
                        print(f"✓ Removed {person_name} from FAISS index")
                    else:
                        print(f"Warning: Failed to remove from FAISS index: {remove_result.get('error')}")
                except Exception as e:
                    print(f"Warning: Failed to remove from FAISS index: {str(e)}")
                    import traceback
                    traceback.print_exc()
        
        return jsonify(result), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ==================== CLOUD WEAPON DETECTION ====================

@app.route('/weapon/detect', methods=['POST'])
def weapon_detect():
    """Detect weapons in an uploaded image using Cloud Verification REST API."""
    import requests
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400

        file = request.files['image']
        if not file.filename:
            return jsonify({'error': 'No file selected'}), 400

        # Retrieve credentials from environment
        api_user = os.getenv('IMAGE_DETECTION_API_USER')
        api_secret = os.getenv('IMAGE_DETECTION_API_SECRET')

        if not api_user or not api_secret:
            return jsonify({
                'success': False,
                'error': 'Verification API credentials (IMAGE_DETECTION_API_USER / IMAGE_DETECTION_API_SECRET) are not configured in backend-service/.env'
            }), 500

        # Read file stream
        file.seek(0)
        image_data = file.read()

        # Prepare multipart payloads
        files = {'media': ('image.jpg', image_data, file.content_type or 'image/jpeg')}
        data = {
            'models': 'weapon',
            'api_user': api_user,
            'api_secret': api_secret
        }

        # Query verification check API
        response = requests.post('https://api.sightengine.com/1.0/check.json', files=files, data=data)

        if response.status_code != 200:
            return jsonify({
                'success': False,
                'error': f"Verification API returned status code {response.status_code}: {response.text}"
            }), 400

        res_data = response.json()
        if res_data.get('status') != 'success':
            error_msg = res_data.get('error', {}).get('message', 'Verification request failed')
            return jsonify({
                'success': False,
                'error': f"Verification API Error: {error_msg}"
            }), 400

        # Extract classification scores
        weapon_info = res_data.get('weapon', {})
        classes = weapon_info.get('classes', {})

        detections = []
        classes_found = set()
        anomalies = []

        # Mapping weapon categories to frontend styles
        class_mapping = {
            'firearm': 'Pistol',
            'knife': 'Knife',
            'firearm_gesture': 'Unknown',
            'firearm_toy': 'Unknown'
        }

        # Detection threshold set at 20%
        DETECTION_THRESHOLD = 0.2

        for se_class, score in classes.items():
            if score >= DETECTION_THRESHOLD:
                frontend_class = class_mapping.get(se_class, 'Unknown')
                conf_percent = round(score * 100, 2)

                # Return placeholder coordinates to satisfy UI and report rendering
                detections.append({
                    'class': frontend_class,
                    'confidence': conf_percent,
                    'bbox': {
                        'x': 50.0,
                        'y': 50.0,
                        'width': 100.0,
                        'height': 100.0
                    }
                })

                if frontend_class != 'Unknown':
                    classes_found.add(frontend_class)
                else:
                    readable_name = se_class.replace('_', ' ').title()
                    classes_found.add(readable_name)

                if score >= 0.8:
                    anomalies.append(f"{frontend_class} ({se_class}) detected with {conf_percent:.1f}% confidence — high certainty threat")
                else:
                    anomalies.append(f"{frontend_class} ({se_class}) detected with {conf_percent:.1f}% confidence")

        # Tag true weapon threats (firearms or knives)
        weapons_found = any(classes.get(w, 0.0) >= DETECTION_THRESHOLD for w in ['firearm', 'knife'])

        raw_result = {
            'model': 'Cloud Verification API (weapon)',
            'total_detections': len(detections),
            'classes': list(classes_found),
            'confidence_threshold': DETECTION_THRESHOLD,
            'raw_api_response': res_data
        }

        return jsonify({
            'success': True,
            'weaponsFound': weapons_found,
            'weaponsDetected': list(classes_found),
            'detections': detections,
            'anomalies': anomalies,
            'totalDetections': len(detections),
            'rawResult': raw_result,
        }), 200

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500



# ==================== HYBRID FORENSIC ANALYSIS (METADATA + ELA) ====================

from PIL import ImageChops, ImageEnhance
from PIL.ExifTags import TAGS
from datetime import datetime as dt
import subprocess
import json as jsonlib

class HybridForensicAnalyzer:
    def __init__(self, image_path):
        self.image_path = image_path
        self.score = 0
        self.reasons = []

    # ---- METADATA EXTRACTION ----
    def extract_metadata(self):
        """Try exiftool first, fall back to PIL EXIF"""
        metadata = {}

        # Try exiftool (richer metadata)
        try:
            result = subprocess.run(
                ["exiftool", "-json", self.image_path],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0 and result.stdout.strip():
                raw = jsonlib.loads(result.stdout)[0]
                # Filter out binary/thumbnail fields and convert values
                skip_keys = ["ThumbnailImage", "PreviewImage", "JFIFThumbnail",
                             "SourceFile", "Directory", "ExifToolVersion"]
                for k, v in raw.items():
                    if k in skip_keys:
                        continue
                    if isinstance(v, (str, int, float, bool)):
                        metadata[k] = v
                    else:
                        metadata[k] = str(v)
                return metadata, "exiftool"
        except (FileNotFoundError, subprocess.TimeoutExpired, Exception) as e:
            print(f"exiftool not available, falling back to PIL: {e}")

        # Fallback: PIL EXIF
        try:
            image = Image.open(self.image_path)
            exif_data = image._getexif()
            if exif_data:
                for tag, value in exif_data.items():
                    decoded = TAGS.get(tag, tag)
                    if isinstance(value, bytes):
                        try:
                            value = value.decode('utf-8', errors='replace')
                        except:
                            value = str(value)
                    elif not isinstance(value, (str, int, float, bool)):
                        value = str(value)
                    metadata[decoded] = value
        except Exception:
            pass

        return metadata, "PIL"

    # ---- METADATA ANALYSIS ----
    def analyze_metadata(self, metadata):
        flags = []

        # Check 1: Camera info
        if "Make" not in metadata and "Camera Make" not in metadata:
            flags.append({"text": "Camera information missing", "severity": "info", "points": 0})

        # Check 2: Editing software
        software = metadata.get("Software", metadata.get("Creator Tool", ""))
        if software:
            editors = ["Adobe", "Photoshop", "GIMP", "Lightroom", "Snapseed", "PicsArt", "Canva", "Figma"]
            if any(tool.lower() in str(software).lower() for tool in editors):
                self.score += 3
                flags.append({"text": f"Edited using: {software}", "severity": "high", "points": 3})
            else:
                flags.append({"text": f"Software: {software}", "severity": "info", "points": 0})

        # Check 3: Missing original date
        has_date = any(k in metadata for k in ["DateTimeOriginal", "Date/Time Original", "CreateDate", "Create Date"])
        if not has_date:
            flags.append({"text": "Missing original capture date", "severity": "info", "points": 0})

        # Check 4: Resolution anomaly
        x_res = metadata.get("XResolution", metadata.get("X Resolution", None))
        if x_res is not None and str(x_res).strip() in ["1", "1.0"]:
            flags.append({"text": "Resolution = 1 (export/web artifact)", "severity": "info", "points": 0})

        # Check 5: GPS data present
        gps_keys = [k for k in metadata if 'GPS' in str(k).upper()]
        if gps_keys:
            flags.append({"text": f"GPS location data found ({len(gps_keys)} fields)", "severity": "info", "points": 0})

        # Check 6: File size vs dimensions check
        file_size = metadata.get("FileSize", metadata.get("File Size", ""))
        if file_size and "KB" in str(file_size).upper():
            # Very small file might indicate heavy compression/editing
            try:
                size_val = float(str(file_size).split()[0])
                if size_val < 50:
                    flags.append({"text": f"Unusually small file size: {file_size}", "severity": "info", "points": 0})
            except:
                pass

        return flags

    # ---- ELA ANALYSIS ----
    def perform_ela(self, quality=90):
        ela_result = {
            "performed": False,
            "meanIntensity": 0,
            "elaImage": None,
            "interpretation": ""
        }

        try:
            original = Image.open(self.image_path).convert("RGB")

            # Save compressed version to temp file
            ela_temp = self.image_path + "_ela_temp.jpg"
            original.save(ela_temp, "JPEG", quality=quality)
            compressed = Image.open(ela_temp)

            # Compute difference
            ela_image = ImageChops.difference(original, compressed)

            extrema = ela_image.getextrema()
            max_diff = max([ex[1] for ex in extrema])
            if max_diff == 0:
                max_diff = 1

            # Compute mean intensity of unscaled differences
            ela_np_unscaled = np.array(ela_image)
            mean_intensity = float(np.mean(ela_np_unscaled))

            scale = 255.0 / max_diff
            ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)

            # Score based on ELA (unscaled mean intensity)
            if mean_intensity > 12.0:
                self.score += 5
                interpretation = "High compression inconsistency — strong indicator of tampering"
                self.reasons.append(f"ELA: High inconsistency (mean={mean_intensity:.1f}) (+5)")
            elif mean_intensity > 6.0:
                self.score += 3
                interpretation = "Moderate compression inconsistency — possible editing detected"
                self.reasons.append(f"ELA: Moderate inconsistency (mean={mean_intensity:.1f}) (+3)")
            else:
                interpretation = "Uniform compression pattern — low suspicion of tampering"
                self.reasons.append(f"ELA: Uniform pattern (mean={mean_intensity:.1f})")

            # Save ELA image to temp and convert to base64
            ela_output_path = self.image_path + "_ela_output.jpg"
            ela_image.save(ela_output_path)

            with open(ela_output_path, "rb") as f:
                ela_base64 = base64.b64encode(f.read()).decode("utf-8")

            ela_result = {
                "performed": True,
                "meanIntensity": round(mean_intensity, 2),
                "elaImage": f"data:image/jpeg;base64,{ela_base64}",
                "interpretation": interpretation
            }

            # Clean up temp files
            for p in [ela_temp, ela_output_path]:
                if os.path.exists(p):
                    os.unlink(p)

        except Exception as e:
            print(f"ELA error: {e}")
            import traceback
            traceback.print_exc()
            ela_result["interpretation"] = f"ELA analysis failed: {str(e)}"

        return ela_result

    # ---- PRNU NOISE RESIDUAL ANALYSIS ----
    def perform_prnu_analysis(self, block_size=64):
        prnu_result = {
            "performed": False,
            "noiseMap": None,
            "blockVariances": [],
            "uniformityScore": 0,
            "suspiciousBlocks": 0,
            "totalBlocks": 0,
            "interpretation": ""
        }

        try:
            import pywt
            from scipy.ndimage import uniform_filter

            img = cv2.imread(self.image_path)
            if img is None:
                prnu_result["interpretation"] = "Could not load image for PRNU analysis"
                return prnu_result

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
            gray = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)

            # --- Wavelet-based noise extraction (inspired by polimi-ispl/prnu-python) ---
            levels = 4
            sigma = 5.0
            noise_var = sigma ** 2

            # Process grayscale channel
            wlet = pywt.wavedec2(gray, 'db4', level=levels)

            # Wiener adaptive filter on detail coefficients
            for level_idx in range(len(wlet) - 1):
                detail_level = wlet[level_idx + 1]
                filtered = []
                for coeff in detail_level:
                    # Adaptive Wiener filter
                    energy = coeff ** 2
                    avg_energy = uniform_filter(energy, size=3, mode='constant')
                    threshold_val = np.maximum(avg_energy - noise_var, 0)
                    filtered_coeff = coeff * noise_var / (threshold_val + noise_var)
                    filtered.append(filtered_coeff)
                wlet[level_idx + 1] = tuple(filtered)

            # Zero out approximation coefficients
            wlet[0] = np.zeros_like(wlet[0])

            # Reconstruct noise residual
            noise_residual = pywt.waverec2(wlet, 'db4')
            noise_residual = noise_residual[:gray.shape[0], :gray.shape[1]]

            # --- Block-wise variance analysis ---
            h, w = noise_residual.shape
            block_h = h // block_size
            block_w = w // block_size

            if block_h < 2 or block_w < 2:
                # Image too small for block analysis
                block_size = min(h, w) // 4
                if block_size < 8:
                    prnu_result["interpretation"] = "Image too small for PRNU block analysis"
                    return prnu_result
                block_h = h // block_size
                block_w = w // block_size

            block_variances = np.zeros((block_h, block_w))
            for i in range(block_h):
                for j in range(block_w):
                    block = noise_residual[
                        i * block_size:(i + 1) * block_size,
                        j * block_size:(j + 1) * block_size
                    ]
                    block_variances[i, j] = np.var(block)

            # Statistical analysis of block variances
            mean_var = np.mean(block_variances)
            std_var = np.std(block_variances)
            total_blocks = block_h * block_w

            # Flag blocks with variance significantly different from mean
            # Add standard deviation floor (20% of mean variance) to avoid false positives in uniform images
            min_increment = 0.2 * mean_var if mean_var > 0 else 0.1
            threshold = mean_var + max(2.0 * std_var, min_increment)
            suspicious_mask = block_variances > threshold
            suspicious_count = int(np.sum(suspicious_mask))
            suspicious_ratio = suspicious_count / total_blocks if total_blocks > 0 else 0

            # Uniformity score (coefficient of variation)
            cv_score = float(std_var / mean_var) if mean_var > 0 else 0

            # Generate heatmap
            variance_norm = block_variances.copy()
            if variance_norm.max() > 0:
                variance_norm = (variance_norm / variance_norm.max() * 255).astype(np.uint8)
            else:
                variance_norm = variance_norm.astype(np.uint8)

            # Resize heatmap to original image size
            heatmap = cv2.resize(variance_norm, (w, h), interpolation=cv2.INTER_NEAREST)
            heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

            # Blend with original image
            original_bgr = cv2.imread(self.image_path)
            original_resized = cv2.resize(original_bgr, (w, h))
            blended = cv2.addWeighted(original_resized, 0.5, heatmap_color, 0.5, 0)

            # Encode to base64
            _, buffer = cv2.imencode('.jpg', blended, [cv2.IMWRITE_JPEG_QUALITY, 85])
            noise_map_b64 = base64.b64encode(buffer).decode('utf-8')

            # Scoring
            if suspicious_ratio > 0.25:
                self.score += 5
                interpretation = "Highly non-uniform noise pattern — strong indicator of splicing or compositing"
                self.reasons.append(f"PRNU: {suspicious_count}/{total_blocks} blocks anomalous ({suspicious_ratio:.0%}) (+5)")
            elif suspicious_ratio > 0.10:
                self.score += 3
                interpretation = "Some non-uniform noise blocks detected — possible local editing"
                self.reasons.append(f"PRNU: {suspicious_count}/{total_blocks} blocks anomalous ({suspicious_ratio:.0%}) (+3)")
            elif cv_score > 1.0:
                self.score += 1
                interpretation = "Slightly irregular noise distribution — minor anomalies detected"
                self.reasons.append(f"PRNU: Noise CV={cv_score:.2f} (+1)")
            else:
                interpretation = "Uniform noise pattern — consistent with an untampered image"
                self.reasons.append(f"PRNU: Uniform pattern (CV={cv_score:.2f})")

            prnu_result = {
                "performed": True,
                "noiseMap": f"data:image/jpeg;base64,{noise_map_b64}",
                "uniformityScore": round(cv_score, 3),
                "suspiciousBlocks": suspicious_count,
                "totalBlocks": total_blocks,
                "suspiciousRatio": round(suspicious_ratio, 4),
                "meanVariance": round(float(mean_var), 4),
                "interpretation": interpretation
            }

        except ImportError:
            prnu_result["interpretation"] = "PRNU analysis requires PyWavelets (pywt). Install with: pip install PyWavelets"
        except Exception as e:
            print(f"PRNU error: {e}")
            import traceback
            traceback.print_exc()
            prnu_result["interpretation"] = f"PRNU analysis failed: {str(e)}"

        return prnu_result

    # ---- FULL ANALYSIS ----
    def run(self):
        metadata, source = self.extract_metadata()
        has_metadata = len(metadata) > 0

        # Metadata analysis
        if has_metadata:
            metadata_flags = self.analyze_metadata(metadata)
        else:
            metadata_flags = [{"text": "No metadata found — may have been stripped", "severity": "info", "points": 0}]

        # ELA analysis
        ela_result = self.perform_ela()

        # PRNU noise analysis
        prnu_result = self.perform_prnu_analysis()

        # Compute verdict
        if self.score <= 3:
            verdict = "Likely Original"
            risk = "LOW"
        elif self.score <= 7:
            verdict = "Possibly Edited"
            risk = "MEDIUM"
        elif self.score <= 12:
            verdict = "Highly Suspicious"
            risk = "HIGH"
        else:
            verdict = "Very High Tampering Probability"
            risk = "CRITICAL"

        return {
            "metadata": metadata,
            "metadataSource": source,
            "hasMetadata": has_metadata,
            "metadataFlags": metadata_flags,
            "ela": ela_result,
            "prnu": prnu_result,
            "score": self.score,
            "maxScore": 24,
            "risk": risk,
            "verdict": verdict,
            "reasons": self.reasons
        }


@app.route('/metadata/analyze', methods=['POST'])
@require_api_key
def analyze_metadata():
    """Hybrid forensic analysis: metadata + ELA"""
    try:
        temp_path = None

        # Handle file upload
        if 'image' in request.files:
            file = request.files['image']
            if file.filename == '':
                return jsonify({'error': 'No file selected'}), 400
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                file.save(tmp_file.name)
                temp_path = tmp_file.name

        # Handle base64 image
        elif request.is_json and 'image' in request.json:
            image_data = request.json['image']
            if ',' in image_data:
                image_data = image_data.split(',', 1)[1]
            image_bytes = base64.b64decode(image_data)
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                tmp_file.write(image_bytes)
                temp_path = tmp_file.name
        else:
            return jsonify({'error': 'No image provided'}), 400

        try:
            analyzer = HybridForensicAnalyzer(temp_path)
            result = analyzer.run()
            return jsonify(result), 200
        finally:
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port, debug=False)
