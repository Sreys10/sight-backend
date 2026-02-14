"""
Face matching module using DeepFace with FAISS for embedding storage
"""
import cv2
from deepface import DeepFace
import os
import numpy as np
import base64
from io import BytesIO
from PIL import Image
import faiss
import pickle
import json
from typing import List, Dict, Optional, Tuple
from datetime import datetime


class FaceMatcher:
    def __init__(self, default_database_path="database/", 
                 faiss_index_path="faiss_index.bin",
                 metadata_path="faiss_metadata.json"):
        self.default_database_path = default_database_path
        self.faiss_index_path = faiss_index_path
        self.metadata_path = metadata_path
        
        # Initialize FAISS index
        self.index = None
        self.metadata = []  # List of dicts: [{"person_name": "...", "image_path": "...", ...}, ...]
        self.embedding_dim = 512  # ArcFace produces 512-dimensional embeddings
        
        # Try to load existing index
        self._load_index()
        
        # Ensure database directory exists
        if not os.path.exists(self.default_database_path):
            os.makedirs(self.default_database_path, exist_ok=True)
    
    def _load_index(self):
        """Load FAISS index and metadata from disk if they exist"""
        try:
            if os.path.exists(self.faiss_index_path) and os.path.exists(self.metadata_path):
                # Load FAISS index
                self.index = faiss.read_index(self.faiss_index_path)
                
                # Load metadata
                with open(self.metadata_path, 'r') as f:
                    self.metadata = json.load(f)
                
                print(f"Loaded FAISS index with {len(self.metadata)} embeddings")
                return True
            else:
                print("No existing FAISS index found. Will create new index when indexing images.")
                return False
        except Exception as e:
            print(f"Error loading FAISS index: {str(e)}")
            self.index = None
            self.metadata = []
            return False
    
    def _save_index(self):
        """Save FAISS index and metadata to disk"""
        try:
            if self.index is not None:
                # Save FAISS index
                faiss.write_index(self.index, self.faiss_index_path)
                
                # Save metadata
                with open(self.metadata_path, 'w') as f:
                    json.dump(self.metadata, f, indent=2)
                
                print(f"Saved FAISS index with {len(self.metadata)} embeddings")
                return True
            return False
        except Exception as e:
            print(f"Error saving FAISS index: {str(e)}")
            return False
    
    def _create_index(self):
        """Create a new FAISS index"""
        # Create L2 (Euclidean) index - we'll use cosine similarity via normalization
        # For cosine similarity, we use InnerProduct index with normalized vectors
        self.index = faiss.IndexFlatIP(self.embedding_dim)  # Inner Product for cosine similarity
        self.metadata = []
        print("Created new FAISS index")
    
    def _normalize_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """Normalize embedding vector for cosine similarity"""
        norm = np.linalg.norm(embedding)
        if norm > 0:
            return embedding / norm
        return embedding
    
    def _generate_embedding(self, img_path: str, model_name: str = "ArcFace", 
                           detector_backend: str = "retinaface") -> Optional[np.ndarray]:
        """
        Generate face embedding for an image
        
        Args:
            img_path: Path to image file
            model_name: Recognition model to use
            detector_backend: Face detector to use
        
        Returns:
            numpy array of embedding or None if face not detected
        """
        try:
            # Use DeepFace to generate embedding
            embedding = DeepFace.represent(
                img_path=img_path,
                model_name=model_name,
                detector_backend=detector_backend,
                enforce_detection=False
            )
            
            if embedding and len(embedding) > 0:
                # Return the first face's embedding
                emb_array = np.array(embedding[0]['embedding'], dtype=np.float32)
                # Normalize for cosine similarity
                return self._normalize_embedding(emb_array)
            return None
        except Exception as e:
            print(f"Error generating embedding: {str(e)}")
            return None
    
    def index_database_images(self, database_path: Optional[str] = None, 
                             model_name: str = "ArcFace", 
                             detector_backend: str = "retinaface") -> Dict:
        """
        Index all images in the database folder to FAISS
        
        Args:
            database_path: Path to database folder
            model_name: Recognition model to use
            detector_backend: Face detector to use
        
        Returns:
            dict with indexing results
        """
        if database_path is None:
            database_path = self.default_database_path
        
        if not os.path.exists(database_path) or not os.path.isdir(database_path):
            return {
                'success': False,
                'error': f'Database path "{database_path}" does not exist',
                'indexed': 0,
                'failed': 0
            }
        
        # Get all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
        image_files = [
            f for f in os.listdir(database_path)
            if os.path.isfile(os.path.join(database_path, f))
            and os.path.splitext(f.lower())[1] in image_extensions
        ]
        
        # Create new index
        self._create_index()
        
        indexed_count = 0
        failed_count = 0
        errors = []
        
        for image_file in image_files:
            image_path = os.path.join(database_path, image_file)
            person_name = os.path.splitext(image_file)[0]
            
            try:
                # Generate embedding
                embedding = self._generate_embedding(
                    image_path, 
                    model_name=model_name,
                    detector_backend=detector_backend
                )
                
                if embedding is None:
                    failed_count += 1
                    errors.append(f"No face detected in {image_file}")
                    continue
                
                # Add to FAISS index
                embedding_2d = embedding.reshape(1, -1)  # Reshape to (1, dim) for FAISS
                self.index.add(embedding_2d)
                
                # Try to get additional metadata from database manager
                additional_metadata = {}
                try:
                    from database_manager import DatabaseManager
                    db_mgr = DatabaseManager()
                    db_person = db_mgr.get_person_by_name(person_name)
                    if db_person:
                        additional_metadata = {
                            'name': db_person.get('name'),
                            'age': db_person.get('age'),
                            'email': db_person.get('email'),
                            'phone': db_person.get('phone'),
                            'notes': db_person.get('notes'),
                            'added_by': db_person.get('added_by'),
                            'added_at': db_person.get('added_at')
                        }
                except:
                    pass
                
                # Store metadata
                self.metadata.append({
                    'person_name': person_name,
                    'image_path': image_path,
                    'model_name': model_name,
                    'detector_backend': detector_backend,
                    'indexed_at': datetime.now().isoformat(),
                    **additional_metadata
                })
                
                indexed_count += 1
                print(f"Indexed: {person_name} ({image_file})")
                
            except Exception as e:
                failed_count += 1
                error_msg = f"Error indexing {image_file}: {str(e)}"
                errors.append(error_msg)
                print(error_msg)
        
        # Save index to disk
        if indexed_count > 0:
            self._save_index()
        
        return {
            'success': True,
            'indexed': indexed_count,
            'failed': failed_count,
            'total': len(image_files),
            'errors': errors[:10]  # Return first 10 errors
        }
    
    def search_embeddings(self, query_embedding: np.ndarray, threshold: float = 0.5,
                        top_k: int = 5) -> List[Dict]:
        """
        Search for similar embeddings in FAISS using cosine similarity
        
        Args:
            query_embedding: numpy array of query embedding (normalized)
            threshold: Minimum similarity score (higher = stricter, range 0-1)
            top_k: Number of top results to return
        
        Returns:
            List of matching documents
        """
        if self.index is None or self.index.ntotal == 0:
            return []
        
        try:
            # Normalize query embedding
            query_embedding = self._normalize_embedding(query_embedding)
            query_2d = query_embedding.reshape(1, -1)
            
            # Search in FAISS (returns cosine similarity scores)
            distances, indices = self.index.search(query_2d, top_k)
            
            results = []
            for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                if idx < len(self.metadata) and distance >= threshold:
                    # Convert cosine similarity to distance (1 - similarity)
                    # For InnerProduct with normalized vectors, higher score = more similar
                    similarity = float(distance)
                    distance_value = 1.0 - similarity
                    
                    # Get full metadata
                    person_meta = self.metadata[idx].copy()
                    
                    results.append({
                        'person_name': person_meta.get('person_name', ''),
                        'image_path': person_meta.get('image_path', ''),
                        'distance': distance_value,
                        'similarity': similarity,
                        'metadata': {
                            'name': person_meta.get('name'),
                            'age': person_meta.get('age'),
                            'email': person_meta.get('email'),
                            'phone': person_meta.get('phone'),
                            'notes': person_meta.get('notes'),
                            'added_by': person_meta.get('added_by'),
                            'added_at': person_meta.get('added_at')
                        }
                    })
            
            return results
            
        except Exception as e:
            print(f"Error searching embeddings: {str(e)}")
            return []
    
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
                    detector_backend="retinaface", threshold=0.5, use_faiss=True):
        """
        Search for a face in the database using FAISS
        
        Args:
            img_path: Path to image file (can contain single or multiple faces)
            database_path: Path to database folder (not used if use_faiss=True)
            model_name: Recognition model (ArcFace, VGG-Face, Facenet, etc.)
            detector_backend: Face detector to use
            threshold: Maximum distance for a match (lower = stricter)
            use_faiss: Whether to use FAISS (True) or DeepFace.find (False)
        
        Returns:
            dict with match results
        """
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
                
                # Save face temporarily for embedding generation
                import tempfile
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as face_tmp:
                    cv2.imwrite(face_tmp.name, cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR))
                    face_path = face_tmp.name
                
                try:
                    match_found = False
                    match_info = None
                    
                    if use_faiss and self.index is not None and self.index.ntotal > 0:
                        # Use FAISS for searching
                        # Generate embedding for the detected face
                        query_embedding = self._generate_embedding(
                            face_path,
                            model_name=model_name,
                            detector_backend=detector_backend
                        )
                        
                        if query_embedding is not None:
                            # Search in FAISS
                            matches = self.search_embeddings(
                                query_embedding,
                                threshold=threshold,
                                top_k=1
                            )
                            
                            if matches and len(matches) > 0:
                                best_match = matches[0]
                                match_found = True
                                
                                # Try to load original image from database metadata
                                original_image_base64 = None
                                try:
                                    from database_manager import DatabaseManager
                                    db_mgr = DatabaseManager()
                                    db_person = db_mgr.get_person_by_name(best_match['person_name'])
                                    if db_person and db_person.get('image_base64'):
                                        original_image_base64 = db_person['image_base64']
                                except:
                                    pass
                                
                                match_info = {
                                    'identity': best_match['image_path'],
                                    'distance': best_match['distance'],
                                    'person_name': best_match['person_name'],
                                    'original_image_base64': original_image_base64,
                                    'metadata': best_match.get('metadata', {})
                                }
                    else:
                        # Fallback to DeepFace.find (original method)
                        if database_path is None:
                            database_path = self.default_database_path
                        
                        if os.path.exists(database_path) and os.path.isdir(database_path):
                            result = DeepFace.find(
                                img_path=face_path,
                                db_path=database_path,
                                model_name=model_name,
                                detector_backend=detector_backend,
                                enforce_detection=False
                            )
                            
                            if len(result) > 0 and len(result[0]) > 0:
                                best = result[0].iloc[0]
                                if best['distance'] <= threshold:
                                    match_found = True
                                    match_info = {
                                        'identity': best['identity'],
                                        'distance': float(best['distance']),
                                        'person_name': os.path.basename(best['identity']).split('.')[0]
                                    }
                        else:
                            match_info = None
                    
                    # Convert face to base64 for response
                    # face_img is already in RGB format from DeepFace, so use it directly
                    if len(face_img.shape) == 3 and face_img.shape[2] == 3:
                        # Already RGB, use directly
                        face_img_display = face_img
                    else:
                        face_img_display = face_img
                    
                    pil_image = Image.fromarray(face_img_display)
                    buffered = BytesIO()
                    pil_image.save(buffered, format="JPEG")
                    img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
                    
                    results.append({
                        'face_number': idx + 1,
                        'match_found': match_found,
                        'match_info': match_info,
                        'face_image_base64': f'data:image/jpeg;base64,{img_base64}'
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
                'matches': results,
                'method': 'faiss' if (use_faiss and self.index is not None and self.index.ntotal > 0) else 'deepface'
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'faces_detected': 0,
                'matches': []
            }
    
    def detect_and_search(self, img_path, database_path=None, model_name="ArcFace",
                         detector_backend="retinaface", threshold=0.5, use_faiss=True):
        """
        Detect faces and search against database in one call
        This is a convenience method that combines detect_faces and search_faces
        """
        return self.search_faces(
            img_path=img_path,
            database_path=database_path,
            model_name=model_name,
            detector_backend=detector_backend,
            threshold=threshold,
            use_faiss=use_faiss
        )
    
    def add_single_image_to_index(self, image_path: str, person_name: str,
                                  model_name: str = "ArcFace",
                                  detector_backend: str = "retinaface",
                                  additional_metadata: Optional[Dict] = None) -> Dict:
        """
        Add a single image to the FAISS index without re-indexing everything
        
        Args:
            image_path: Path to the image file
            person_name: Name/identifier of the person
            model_name: Recognition model to use
            detector_backend: Face detector to use
            additional_metadata: Optional dict with additional metadata (name, age, email, etc.)
        
        Returns:
            dict with indexing result
        """
        try:
            # Initialize index if it doesn't exist
            if self.index is None:
                self._create_index()
            
            # Check if person already exists in index
            existing_index = None
            for idx, meta in enumerate(self.metadata):
                if meta.get('person_name') == person_name:
                    existing_index = idx
                    break
            
            # Generate embedding
            embedding = self._generate_embedding(
                image_path,
                model_name=model_name,
                detector_backend=detector_backend
            )
            
            if embedding is None:
                return {
                    'success': False,
                    'error': 'No face detected in image',
                    'person_name': person_name
                }
            
            # Prepare metadata
            metadata_entry = {
                'person_name': person_name,
                'image_path': image_path,
                'model_name': model_name,
                'detector_backend': detector_backend,
                'indexed_at': datetime.now().isoformat()
            }
            
            if additional_metadata:
                metadata_entry.update(additional_metadata)
            
            if existing_index is not None:
                # Update existing entry
                # FAISS doesn't support direct updates, so we add the new embedding
                # and update metadata. The old embedding will remain but won't match
                # as well. For a proper update, we'd need to rebuild, but that's expensive.
                # For now, we'll just add the new one (duplicate) and update metadata.
                print(f"Updating existing index entry for {person_name} (adding new embedding)")
                
                # Update metadata entry
                self.metadata[existing_index] = metadata_entry
                
                # Add new embedding (old one stays but won't be used in search due to metadata)
                embedding_2d = embedding.reshape(1, -1)
                self.index.add(embedding_2d)
            else:
                # Add new entry
                embedding_2d = embedding.reshape(1, -1)
                self.index.add(embedding_2d)
                self.metadata.append(metadata_entry)
            
            # Save index to disk
            self._save_index()
            
            return {
                'success': True,
                'person_name': person_name,
                'action': 'updated' if existing_index is not None else 'added',
                'total_embeddings': self.index.ntotal
            }
            
        except Exception as e:
            print(f"Error adding image to index: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'error': str(e),
                'person_name': person_name
            }
    
    def remove_from_index(self, person_name: str) -> Dict:
        """
        Remove a person from the FAISS index
        
        Args:
            person_name: Name/identifier of the person to remove
        
        Returns:
            dict with removal result
        """
        try:
            if self.index is None or len(self.metadata) == 0:
                return {
                    'success': False,
                    'error': 'Index is empty',
                    'person_name': person_name
                }
            
            # Find the entry
            found_index = None
            for idx, meta in enumerate(self.metadata):
                if meta.get('person_name') == person_name:
                    found_index = idx
                    break
            
            if found_index is None:
                return {
                    'success': False,
                    'error': f'Person {person_name} not found in index',
                    'person_name': person_name
                }
            
            # Remove from metadata
            self.metadata.pop(found_index)
            
            # Rebuild index without this entry
            # FAISS doesn't support direct removal, so we rebuild
            old_index = self.index
            self._create_index()
            
            # Re-add all embeddings except the removed one
            for idx, meta in enumerate(self.metadata):
                img_path = meta.get('image_path')
                if img_path and os.path.exists(img_path):
                    emb = self._generate_embedding(
                        img_path,
                        model_name=meta.get('model_name', 'ArcFace'),
                        detector_backend=meta.get('detector_backend', 'retinaface')
                    )
                    if emb is not None:
                        emb_2d = emb.reshape(1, -1)
                        self.index.add(emb_2d)
            
            # Save index to disk
            self._save_index()
            
            return {
                'success': True,
                'person_name': person_name,
                'total_embeddings': self.index.ntotal
            }
            
        except Exception as e:
            print(f"Error removing from index: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'person_name': person_name
            }
    
    def get_index_stats(self) -> Dict:
        """Get statistics about the FAISS index"""
        if self.index is None:
            return {
                'indexed': False,
                'total_embeddings': 0,
                'index_path': self.faiss_index_path,
                'metadata_path': self.metadata_path
            }
        
        return {
            'indexed': True,
            'total_embeddings': self.index.ntotal,
            'embedding_dim': self.embedding_dim,
            'index_path': self.faiss_index_path,
            'metadata_path': self.metadata_path,
            'metadata_count': len(self.metadata)
        }
