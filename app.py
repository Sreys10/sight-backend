from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import tempfile
import base64
from io import BytesIO
from PIL import Image
import cv2
import numpy as np

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, use system environment variables

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


# Database management endpoints
try:
    from database_manager import DatabaseManager
    mongodb_uri = os.getenv('MONGODB_URI', 'mongodb://localhost:27017/')
    
    # Check if MongoDB URI is set
    if not mongodb_uri or mongodb_uri == 'mongodb://localhost:27017/':
        print("⚠️  Warning: Using default MongoDB URI. Set MONGODB_URI environment variable for MongoDB Atlas.")
        print("   For MongoDB Atlas setup, see MONGODB_ATLAS_SETUP.md")
    
    db_manager = DatabaseManager(mongodb_uri=mongodb_uri)
    DATABASE_MANAGER_AVAILABLE = True
    
    # Check if connection was successful
    if db_manager.collection is None:
        print("⚠️  Warning: Database manager initialized but MongoDB connection failed.")
        print("   Face database features will not be available.")
        print("   Please check your MONGODB_URI and MongoDB connection.")
except ImportError:
    print("Warning: Database manager not available.")
    DATABASE_MANAGER_AVAILABLE = False
    db_manager = None


@app.route('/face/database/list', methods=['GET'])
def list_database():
    """List all persons in the database"""
    if not DATABASE_MANAGER_AVAILABLE:
        return jsonify({'error': 'Database manager not available'}), 503
    
    try:
        persons = db_manager.get_all_persons()
        return jsonify({
            'success': True,
            'persons': persons
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/face/database/add', methods=['POST'])
def add_to_database():
    """Add a person to the database"""
    if not DATABASE_MANAGER_AVAILABLE:
        return jsonify({'error': 'Database manager not available'}), 503
    
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
        
        result = db_manager.add_person(
            image_file=file,
            person_name=person_name,
            name=request.form.get('name', ''),
            age=request.form.get('age'),
            email=request.form.get('email', ''),
            phone=request.form.get('phone', ''),
            notes=request.form.get('notes', ''),
            added_by_name=request.form.get('added_by_name', ''),
            added_by_email=request.form.get('added_by_email', '')
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
                    local_path = os.path.join(db_manager.database_path, f"{person_name}{image_ext}")
                    
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
def update_person(person_id):
    """Update person metadata"""
    if not DATABASE_MANAGER_AVAILABLE:
        return jsonify({'error': 'Database manager not available'}), 503
    
    try:
        data = request.json
        result = db_manager.update_person(
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
def delete_person(person_id):
    """Delete a person from the database"""
    if not DATABASE_MANAGER_AVAILABLE:
        return jsonify({'error': 'Database manager not available'}), 503
    
    try:
        # Get person_name before deletion (needed for FAISS index removal)
        person_name = None
        if FACE_MATCHING_AVAILABLE:
            try:
                person = db_manager.get_person(person_id)
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
        
        result = db_manager.delete_person(person_id)
        
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
            self.score += 3
            flags.append({"text": "Camera information missing", "severity": "high", "points": 3})

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
            self.score += 2
            flags.append({"text": "Missing original capture date", "severity": "medium", "points": 2})

        # Check 4: Resolution anomaly
        x_res = metadata.get("XResolution", metadata.get("X Resolution", None))
        if x_res is not None and str(x_res).strip() in ["1", "1.0"]:
            self.score += 1
            flags.append({"text": "Resolution = 1 (export/web artifact)", "severity": "low", "points": 1})

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
                    self.score += 1
                    flags.append({"text": f"Unusually small file size: {file_size}", "severity": "low", "points": 1})
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

            scale = 255.0 / max_diff
            ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)

            # Compute mean intensity
            ela_np = np.array(ela_image)
            mean_intensity = float(np.mean(ela_np))

            # Score based on ELA
            if mean_intensity > 40:
                self.score += 5
                interpretation = "High compression inconsistency — strong indicator of tampering"
                self.reasons.append(f"ELA: High inconsistency (mean={mean_intensity:.1f}) (+5)")
            elif mean_intensity > 25:
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
            threshold = mean_var + 2.0 * std_var
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
            self.score += 3
            metadata_flags = [{"text": "No metadata found — may have been stripped", "severity": "high", "points": 3}]

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
