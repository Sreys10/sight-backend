"""
Database manager for face recognition database with MongoDB storage
"""
import os
from datetime import datetime
from typing import Dict, List, Optional
import base64
from io import BytesIO
from PIL import Image
import cv2
from pymongo import MongoClient
from bson import ObjectId


class DatabaseManager:
    def __init__(self, database_path="database/", mongodb_uri=None):
        self.database_path = database_path
        self.mongodb_uri = mongodb_uri or os.getenv('MONGODB_URI', 'mongodb://localhost:27017/')
        self.db_name = os.getenv('MONGODB_DB_NAME', 'evi-check')
        
        # Connect to MongoDB
        try:
            # For MongoDB Atlas, use serverSelectionTimeoutMS
            if 'mongodb+srv://' in self.mongodb_uri or 'mongodb.net' in self.mongodb_uri:
                # MongoDB Atlas connection
                self.client = MongoClient(
                    self.mongodb_uri,
                    serverSelectionTimeoutMS=5000,
                    connectTimeoutMS=10000,
                    socketTimeoutMS=10000
                )
                print(f"Connecting to MongoDB Atlas: {self.db_name}")
            else:
                # Local MongoDB connection
                self.client = MongoClient(
                    self.mongodb_uri,
                    serverSelectionTimeoutMS=5000
                )
                print(f"Connecting to local MongoDB: {self.db_name}")
            
            # Test connection
            self.client.admin.command('ping')
            self.db = self.client[self.db_name]
            self.collection = self.db['face_database']
            print(f"✓ Successfully connected to MongoDB: {self.db_name}")
        except Exception as e:
            print(f"✗ MongoDB connection failed: {str(e)}")
            print(f"  URI: {self.mongodb_uri[:50]}..." if len(self.mongodb_uri) > 50 else f"  URI: {self.mongodb_uri}")
            print("  Please check:")
            print("  1. MONGODB_URI environment variable is set correctly")
            print("  2. MongoDB Atlas network access allows your IP (0.0.0.0/0 for all)")
            print("  3. Database username and password are correct")
            print("  4. Connection string format: mongodb+srv://username:password@cluster.mongodb.net/")
            self.client = None
            self.db = None
            self.collection = None
        
        # Ensure database directory exists (for temporary storage during processing)
        if not os.path.exists(self.database_path):
            os.makedirs(self.database_path, exist_ok=True)
    
    def _image_to_base64(self, image_path: str) -> Optional[str]:
        """Convert image file to base64 string"""
        try:
            img = cv2.imread(image_path)
            if img is not None:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(img_rgb)
                buffered = BytesIO()
                pil_img.save(buffered, format="JPEG")
                img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
                return f'data:image/jpeg;base64,{img_base64}'
        except Exception as e:
            print(f"Error converting image to base64: {str(e)}")
        return None
    
    def _file_to_base64(self, file) -> Optional[str]:
        """Convert file object to base64 string"""
        try:
            if hasattr(file, 'read'):
                # File object (Flask file object or file-like)
                file.seek(0)
                file_bytes = file.read()
                if not file_bytes:
                    print("Warning: File read returned empty bytes")
                    return None
                img_base64 = base64.b64encode(file_bytes).decode('utf-8')
                return f'data:image/jpeg;base64,{img_base64}'
            elif isinstance(file, str) and os.path.exists(file):
                # File path
                return self._image_to_base64(file)
            else:
                print(f"Warning: Unsupported file type: {type(file)}")
                return None
        except Exception as e:
            print(f"Error converting file to base64: {str(e)}")
            import traceback
            traceback.print_exc()
        return None
    
    def add_person(self, image_file, person_name: str, name: str = "", age: int = None,
                   email: str = "", phone: str = "", notes: str = "",
                   added_by_name: str = "", added_by_email: str = "") -> Dict:
        """
        Add a person to the database (stored in MongoDB)
        """
        if self.collection is None:
            return {'success': False, 'error': 'MongoDB not available'}
        
        try:
            # Convert image to base64
            image_base64 = self._file_to_base64(image_file)
            if not image_base64:
                return {'success': False, 'error': 'Failed to process image'}
            
            # Generate unique ID
            person_id = f"{person_name}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            
            # Determine file extension for reference
            if hasattr(image_file, 'filename'):
                filename = image_file.filename
                ext = os.path.splitext(filename)[1] or '.jpg'
            else:
                ext = os.path.splitext(str(image_file))[1] or '.jpg'
            
            # Create document
            person_doc = {
                'id': person_id,
                'person_name': person_name,
                'image_base64': image_base64,
                'image_extension': ext,
                'name': name,
                'age': int(age) if age else None,
                'email': email,
                'phone': phone,
                'notes': notes,
                'added_by': {
                    'name': added_by_name,
                    'email': added_by_email
                } if added_by_name else None,
                'added_at': datetime.now().isoformat(),
                'created_at': datetime.now()
            }
            
            # Insert into MongoDB
            result = self.collection.insert_one(person_doc)
            
            # Also save a local copy for FAISS indexing (temporary)
            local_path = os.path.join(self.database_path, f"{person_name}{ext}")
            try:
                if hasattr(image_file, 'read'):
                    # Flask file object or file-like object
                    image_file.seek(0)
                    with open(local_path, 'wb') as f:
                        f.write(image_file.read())
                elif isinstance(image_file, str) and os.path.exists(image_file):
                    # File path string
                    import shutil
                    shutil.copy2(image_file, local_path)
            except Exception as e:
                print(f"Warning: Failed to save local copy for FAISS indexing: {str(e)}")
                # Continue anyway - MongoDB storage is the primary storage
            
            return {
                'success': True,
                'person': {
                    'id': person_id,
                    'person_name': person_name,
                    'image_base64': image_base64,
                    'name': name,
                    'age': int(age) if age else None,
                    'email': email,
                    'phone': phone,
                    'notes': notes,
                    'added_by': person_doc['added_by'],
                    'added_at': person_doc['added_at']
                }
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def get_all_persons(self) -> List[Dict]:
        """Get all persons from MongoDB"""
        if self.collection is None:
            return []
        
        try:
            persons = list(self.collection.find({}, {'_id': 0}))
            # Convert ObjectId to string if present
            for person in persons:
                if '_id' in person:
                    person['_id'] = str(person['_id'])
            return persons
        except Exception as e:
            print(f"Error getting all persons: {str(e)}")
            return []
    
    def get_person(self, person_id: str) -> Optional[Dict]:
        """Get person by ID"""
        if self.collection is None:
            return None
        
        try:
            person = self.collection.find_one({'id': person_id}, {'_id': 0})
            if person and '_id' in person:
                person['_id'] = str(person['_id'])
            return person
        except Exception as e:
            print(f"Error getting person: {str(e)}")
            return None
    
    def update_person(self, person_id: str, **kwargs) -> Dict:
        """Update person metadata"""
        if self.collection is None:
            return {'success': False, 'error': 'MongoDB not available'}
        
        try:
            update_data = {}
            for key, value in kwargs.items():
                if key == 'age' and value:
                    update_data[key] = int(value)
                elif key in ['name', 'email', 'phone', 'notes', 'person_name']:
                    update_data[key] = value
            
            if not update_data:
                return {'success': False, 'error': 'No fields to update'}
            
            result = self.collection.update_one(
                {'id': person_id},
                {'$set': update_data}
            )
            
            if result.matched_count == 0:
                return {'success': False, 'error': 'Person not found'}
            
            # Return updated person
            person = self.get_person(person_id)
            return {
                'success': True,
                'person': person
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def delete_person(self, person_id: str) -> Dict:
        """Delete person from MongoDB"""
        if self.collection is None:
            return {'success': False, 'error': 'MongoDB not available'}
        
        try:
            # Get person first to delete local file if exists
            person = self.get_person(person_id)
            if person:
                # Delete local file if exists
                local_path = os.path.join(self.database_path, f"{person.get('person_name', '')}{person.get('image_extension', '.jpg')}")
                if os.path.exists(local_path):
                    try:
                        os.remove(local_path)
                    except:
                        pass
            
            result = self.collection.delete_one({'id': person_id})
            
            if result.deleted_count == 0:
                return {'success': False, 'error': 'Person not found'}
            
            return {'success': True}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def get_person_by_name(self, person_name: str) -> Optional[Dict]:
        """Get person by person_name"""
        if self.collection is None:
            return None
        
        try:
            person = self.collection.find_one({'person_name': person_name}, {'_id': 0})
            if person and '_id' in person:
                person['_id'] = str(person['_id'])
            return person
        except Exception as e:
            print(f"Error getting person by name: {str(e)}")
            return None
