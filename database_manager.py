"""
Database manager for face recognition database with PostgreSQL storage
"""
import os
from datetime import datetime
from typing import Dict, List, Optional
import base64
from io import BytesIO
from PIL import Image
import cv2
import psycopg2
import psycopg2.extras
import json


class DatabaseManager:
    def __init__(self, database_path="database/", database_url=None):
        self.database_path = database_path
        self.database_url = database_url or os.getenv('DATABASE_URL')

        # Connect to PostgreSQL
        self.conn = None
        self.cursor = None
        try:
            if not self.database_url:
                raise ValueError("DATABASE_URL environment variable is not set")

            self.conn = psycopg2.connect(self.database_url)
            self.conn.autocommit = True
            self.cursor = self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

            # Create table if it doesn't exist
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS face_database (
                    id          TEXT PRIMARY KEY,
                    person_name TEXT NOT NULL,
                    image_base64 TEXT,
                    image_extension TEXT DEFAULT '.jpg',
                    name        TEXT,
                    age         INTEGER,
                    email       TEXT,
                    phone       TEXT,
                    notes       TEXT,
                    added_by    JSONB,
                    added_at    TEXT,
                    created_at  TIMESTAMPTZ DEFAULT NOW()
                )
            """)
            print(f"✓ Successfully connected to PostgreSQL face database")

        except Exception as e:
            print(f"✗ PostgreSQL connection failed: {str(e)}")
            print("  Please check your DATABASE_URL environment variable.")
            self.conn = None
            self.cursor = None

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
                file.seek(0)
                file_bytes = file.read()
                if not file_bytes:
                    print("Warning: File read returned empty bytes")
                    return None
                img_base64 = base64.b64encode(file_bytes).decode('utf-8')
                return f'data:image/jpeg;base64,{img_base64}'
            elif isinstance(file, str) and os.path.exists(file):
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
        """Add a person to the database (stored in PostgreSQL)"""
        if self.cursor is None:
            return {'success': False, 'error': 'PostgreSQL not available'}

        try:
            # Convert image to base64
            image_base64 = self._file_to_base64(image_file)
            if not image_base64:
                return {'success': False, 'error': 'Failed to process image'}

            # Generate unique ID
            person_id = f"{person_name}_{datetime.now().strftime('%Y%m%d%H%M%S')}"

            # Determine file extension
            if hasattr(image_file, 'filename'):
                ext = os.path.splitext(image_file.filename)[1] or '.jpg'
            else:
                ext = os.path.splitext(str(image_file))[1] or '.jpg'

            added_by = json.dumps({'name': added_by_name, 'email': added_by_email}) if added_by_name else None
            added_at = datetime.now().isoformat()

            self.cursor.execute("""
                INSERT INTO face_database
                    (id, person_name, image_base64, image_extension, name, age, email, phone, notes, added_by, added_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                person_id, person_name, image_base64, ext, name,
                int(age) if age else None, email, phone, notes,
                added_by, added_at
            ))

            # Also save a local copy for FAISS indexing (temporary)
            local_path = os.path.join(self.database_path, f"{person_name}{ext}")
            try:
                if hasattr(image_file, 'read'):
                    image_file.seek(0)
                    with open(local_path, 'wb') as f:
                        f.write(image_file.read())
                elif isinstance(image_file, str) and os.path.exists(image_file):
                    import shutil
                    shutil.copy2(image_file, local_path)
            except Exception as e:
                print(f"Warning: Failed to save local copy for FAISS indexing: {str(e)}")

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
                    'added_by': json.loads(added_by) if added_by else None,
                    'added_at': added_at
                }
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def get_all_persons(self) -> List[Dict]:
        """Get all persons from PostgreSQL"""
        if self.cursor is None:
            return []

        try:
            self.cursor.execute("SELECT * FROM face_database ORDER BY created_at DESC")
            rows = self.cursor.fetchall()
            persons = []
            for row in rows:
                p = dict(row)
                if p.get('added_by') and isinstance(p['added_by'], str):
                    p['added_by'] = json.loads(p['added_by'])
                persons.append(p)
            return persons
        except Exception as e:
            print(f"Error getting all persons: {str(e)}")
            return []

    def get_person(self, person_id: str) -> Optional[Dict]:
        """Get person by ID"""
        if self.cursor is None:
            return None

        try:
            self.cursor.execute("SELECT * FROM face_database WHERE id = %s", (person_id,))
            row = self.cursor.fetchone()
            if row:
                p = dict(row)
                if p.get('added_by') and isinstance(p['added_by'], str):
                    p['added_by'] = json.loads(p['added_by'])
                return p
            return None
        except Exception as e:
            print(f"Error getting person: {str(e)}")
            return None

    def update_person(self, person_id: str, **kwargs) -> Dict:
        """Update person metadata"""
        if self.cursor is None:
            return {'success': False, 'error': 'PostgreSQL not available'}

        try:
            allowed = ['name', 'age', 'email', 'phone', 'notes', 'person_name']
            update_data = {}
            for key, value in kwargs.items():
                if key in allowed:
                    if key == 'age' and value:
                        update_data[key] = int(value)
                    else:
                        update_data[key] = value

            if not update_data:
                return {'success': False, 'error': 'No fields to update'}

            set_clause = ", ".join(f"{k} = %s" for k in update_data.keys())
            values = list(update_data.values()) + [person_id]

            self.cursor.execute(
                f"UPDATE face_database SET {set_clause} WHERE id = %s RETURNING id",
                values
            )
            if self.cursor.fetchone() is None:
                return {'success': False, 'error': 'Person not found'}

            person = self.get_person(person_id)
            return {'success': True, 'person': person}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def delete_person(self, person_id: str) -> Dict:
        """Delete person from PostgreSQL"""
        if self.cursor is None:
            return {'success': False, 'error': 'PostgreSQL not available'}

        try:
            person = self.get_person(person_id)
            if person:
                # Delete local file if exists
                local_path = os.path.join(
                    self.database_path,
                    f"{person.get('person_name', '')}{person.get('image_extension', '.jpg')}"
                )
                if os.path.exists(local_path):
                    try:
                        os.remove(local_path)
                    except Exception:
                        pass

            self.cursor.execute(
                "DELETE FROM face_database WHERE id = %s RETURNING id",
                (person_id,)
            )
            if self.cursor.fetchone() is None:
                return {'success': False, 'error': 'Person not found'}

            return {'success': True}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def get_person_by_name(self, person_name: str) -> Optional[Dict]:
        """Get person by person_name"""
        if self.cursor is None:
            return None

        try:
            self.cursor.execute(
                "SELECT * FROM face_database WHERE person_name = %s LIMIT 1",
                (person_name,)
            )
            row = self.cursor.fetchone()
            if row:
                p = dict(row)
                if p.get('added_by') and isinstance(p['added_by'], str):
                    p['added_by'] = json.loads(p['added_by'])
                return p
            return None
        except Exception as e:
            print(f"Error getting person by name: {str(e)}")
            return None
