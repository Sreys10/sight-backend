"""
Face matching module using DeepFace
"""
import cv2
from deepface import DeepFace
import os
import numpy as np
import base64
from io import BytesIO
from PIL import Image


class FaceMatcher:
    def __init__(self, default_database_path="database/"):
        self.default_database_path = default_database_path
        # Ensure database directory exists
        if not os.path.exists(self.default_database_path):
            os.makedirs(self.default_database_path, exist_ok=True)
    
    def detect_faces(self, img_path, detector_backend="retinaface"):
        """
        Detect all faces in an image
        
        Args:
            img_path: Path to image file
            detector_backend: Face detector to use (retinaface, opencv, ssd, dlib, mtcnn)
        
        Returns:
            dict with faces array containing face data and base64 encoded images
        """
        try:
            # Extract all faces
            faces = DeepFace.extract_faces(
                img_path=img_path,
                detector_backend=detector_backend,
                enforce_detection=False
            )
            
            detected_faces = []
            for idx, face in enumerate(faces):
                face_img = face["face"]
                
                # Convert to uint8 if needed
                if face_img.dtype != 'uint8':
                    if face_img.max() <= 1.0:
                        face_img = (face_img * 255).astype('uint8')
                    else:
                        face_img = face_img.astype('uint8')
                
                # Convert to RGB if needed
                if len(face_img.shape) == 3 and face_img.shape[2] == 3:
                    face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                else:
                    face_img_rgb = face_img
                
                # Convert to base64 for JSON response
                pil_image = Image.fromarray(face_img_rgb)
                buffered = BytesIO()
                pil_image.save(buffered, format="JPEG")
                img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
                
                detected_faces.append({
                    'face_number': idx + 1,
                    'image_base64': f'data:image/jpeg;base64,{img_base64}',
                    'shape': face_img_rgb.shape
                })
            
            return {
                'success': True,
                'faces_detected': len(detected_faces),
                'faces': detected_faces
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'faces_detected': 0,
                'faces': []
            }
    
    def search_faces(self, img_path, database_path=None, model_name="ArcFace", 
                    detector_backend="retinaface", threshold=0.5):
        """
        Search for a face in the database
        
        Args:
            img_path: Path to image file (can contain single or multiple faces)
            database_path: Path to database folder (default: self.default_database_path)
            model_name: Recognition model (ArcFace, VGG-Face, Facenet, etc.)
            detector_backend: Face detector to use
            threshold: Maximum distance for a match (lower = stricter)
        
        Returns:
            dict with match results
        """
        if database_path is None:
            database_path = self.default_database_path
        
        try:
            # First detect faces in the input image
            faces = DeepFace.extract_faces(
                img_path=img_path,
                detector_backend=detector_backend,
                enforce_detection=False
            )
            
            if len(faces) == 0:
                return {
                    'success': True,
                    'faces_detected': 0,
                    'matches': []
                }
            
            results = []
            
            # Process each detected face
            for idx, face in enumerate(faces):
                face_img = face["face"]
                
                # Convert to uint8 if needed
                if face_img.dtype != 'uint8':
                    if face_img.max() <= 1.0:
                        face_img = (face_img * 255).astype('uint8')
                    else:
                        face_img = face_img.astype('uint8')
                
                # Save face temporarily for matching
                import tempfile
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as face_tmp:
                    cv2.imwrite(face_tmp.name, cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR))
                    face_path = face_tmp.name
                
                try:
                    # Search in database
                    if os.path.exists(database_path) and os.path.isdir(database_path):
                        result = DeepFace.find(
                            img_path=face_path,
                            db_path=database_path,
                            model_name=model_name,
                            detector_backend=detector_backend,
                            enforce_detection=False
                        )
                        
                        match_found = False
                        match_info = None
                        
                        if len(result) > 0 and len(result[0]) > 0:
                            best = result[0].iloc[0]
                            if best['distance'] <= threshold:
                                match_found = True
                                match_info = {
                                    'identity': best['identity'],
                                    'distance': float(best['distance']),
                                    'person_name': os.path.basename(best['identity']).split('.')[0]
                                }
                        
                        # Convert face to base64 for response
                        face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR)
                        pil_image = Image.fromarray(face_img_rgb)
                        buffered = BytesIO()
                        pil_image.save(buffered, format="JPEG")
                        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
                        
                        results.append({
                            'face_number': idx + 1,
                            'match_found': match_found,
                            'match_info': match_info,
                            'face_image_base64': f'data:image/jpeg;base64,{img_base64}'
                        })
                    else:
                        # No database found
                        face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR)
                        pil_image = Image.fromarray(face_img_rgb)
                        buffered = BytesIO()
                        pil_image.save(buffered, format="JPEG")
                        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
                        
                        results.append({
                            'face_number': idx + 1,
                            'match_found': False,
                            'match_info': None,
                            'face_image_base64': f'data:image/jpeg;base64,{img_base64}',
                            'error': f'Database path "{database_path}" does not exist'
                        })
                finally:
                    # Clean up temp file
                    try:
                        os.unlink(face_path)
                    except:
                        pass
            
            return {
                'success': True,
                'faces_detected': len(results),
                'matches': results
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'faces_detected': 0,
                'matches': []
            }
    
    def detect_and_search(self, img_path, database_path=None, model_name="ArcFace",
                         detector_backend="retinaface", threshold=0.5):
        """
        Detect faces and search against database in one call
        This is a convenience method that combines detect_faces and search_faces
        """
        return self.search_faces(
            img_path=img_path,
            database_path=database_path,
            model_name=model_name,
            detector_backend=detector_backend,
            threshold=threshold
        )

