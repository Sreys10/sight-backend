import os
import time
import uuid
import logging
import numpy as np
import cv2
from typing import List, Optional
from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException, status, Request
from sqlalchemy import text
from sqlalchemy.orm import Session
from database import get_db
from models import Person, FaceEmbedding
from schemas import FaceSearchMatch, MatchedPerson
from services.face_detection import detect_faces
from services.face_embedding import extract_face_embedding
from services.face_matching import search_face_in_db

logger = logging.getLogger("face_recognition")

router = APIRouter(prefix="/api/faces", tags=["Face Recognition"])

# Configuration
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
STORAGE_DIR = "storage"
FACES_DIR = os.path.join(STORAGE_DIR, "faces")

# Ensure storage directories exist
os.makedirs(FACES_DIR, exist_ok=True)

def validate_upload(file: UploadFile) -> str:
    """Validate file size and extension. Returns file extension."""
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported format. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    return ext

async def read_image_from_upload(file: UploadFile) -> np.ndarray:
    """Read upload file bytes and decode into OpenCV BGR image."""
    try:
        contents = await file.read()
        if len(contents) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File exceeds maximum size of {MAX_FILE_SIZE // (1024 * 1024)}MB."
            )
        
        # Reset file cursor just in case
        await file.seek(0)
        
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Corrupted image or invalid format.")
        return img
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error reading image: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Corrupted image or unsupported format. Detail: {str(e)}"
        )


@router.post("/register", status_code=status.HTTP_201_CREATED)
async def register_person(
    request: Request,
    full_name: str = Form(..., description="Full name of the person"),
    gender: Optional[str] = Form(None),
    age: Optional[int] = Form(None),
    case_number: Optional[str] = Form(None),
    notes: Optional[str] = Form(None),
    images: List[UploadFile] = File(..., description="One or multiple portrait images"),
    db: Session = Depends(get_db)
):
    """
    Register a person into the database with one or multiple images containing their face.
    Generates and stores 512-dimensional ArcFace embeddings for each face.
    """
    start_time = time.time()
    
    if not images or len(images) == 0:
        raise HTTPException(status_code=400, detail="At least one image must be provided.")
        
    logger.info(f"Received registration request for full_name='{full_name}', case_number='{case_number}'")
    
    # Check if database is available
    try:
        db.execute(text("SELECT 1"))
    except Exception as db_err:
        logger.error(f"Database unavailable: {str(db_err)}")
        raise HTTPException(status_code=503, detail="Database unavailable. Please check backend connection.")

    # Retrieve InsightFace app from state
    face_app = getattr(request.app.state, "face_analysis", None)
    if face_app is None:
        raise HTTPException(status_code=500, detail="Face recognition model not loaded on server.")

    # Look up existing person or create new to prevent duplicates
    clean_name = full_name.strip()
    clean_case = case_number.strip() if case_number else None
    
    person = db.query(Person).filter(
        Person.full_name == clean_name,
        Person.case_number == clean_case
    ).first()
    
    # Duplicate person check
    if person is not None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Person '{clean_name}' associated with Case '{clean_case}' is already registered."
        )

    # Create new Person record
    person = Person(
        full_name=clean_name,
        gender=gender.strip() if gender else None,
        age=age,
        case_number=clean_case,
        notes=notes.strip() if notes else None
    )
    db.add(person)
    db.flush()  # Populates person.id
    logger.info(f"Created new Person record with ID={person.id}")

    embeddings_created = 0
    saved_files = []
    
    try:
        # Process each uploaded image
        for idx, upload_file in enumerate(images):
            ext = validate_upload(upload_file)
            img = await read_image_from_upload(upload_file)
            
            # Detect faces using InsightFace
            inference_start = time.time()
            faces = detect_faces(face_app, img)
            inference_time = (time.time() - inference_start) * 1000
            logger.info(f"Image {idx+1} detected {len(faces)} face(s) in {inference_time:.1f}ms")
            
            # Constraints: exactly 1 face during registration
            if len(faces) == 0:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"No face found in image {idx+1} ({upload_file.filename})."
                )
            elif len(faces) > 1:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Multiple faces found in image {idx+1} ({upload_file.filename}). Registration photos must contain exactly one face."
                )
                
            # Extract normalized embedding
            embedding = extract_face_embedding(faces[0])
            
            # Save the file to unique name in /storage/faces/
            unique_filename = f"{person.id}_{uuid.uuid4().hex}{ext}"
            file_dest = os.path.join(FACES_DIR, unique_filename)
            
            # Write image to disk
            cv2.imwrite(file_dest, img)
            saved_files.append(file_dest)
            
            # Create FaceEmbedding record
            relative_path = f"faces/{unique_filename}"
            face_embedding = FaceEmbedding(
                person_id=person.id,
                image_path=relative_path,
                embedding=embedding
            )
            db.add(face_embedding)
            embeddings_created += 1

        db.commit()
        total_time = (time.time() - start_time) * 1000
        logger.info(f"Successfully registered {person.full_name}. Created {embeddings_created} embedding(s) in {total_time:.1f}ms")
        
        return {
            "success": True,
            "person_id": str(person.id),
            "embeddings_created": embeddings_created
        }
        
    except Exception as exc:
        db.rollback()
        # Clean up files on failure
        for f in saved_files:
            try:
                os.remove(f)
            except:
                pass
        if isinstance(exc, HTTPException):
            raise exc
        logger.error(f"Unexpected error during registration: {str(exc)}")
        raise HTTPException(status_code=500, detail=f"Registration failed: {str(exc)}")


@router.post("/search", response_model=List[FaceSearchMatch])
async def search_faces_endpoint(
    request: Request,
    image: UploadFile = File(..., description="Image containing one or multiple faces"),
    threshold: float = Form(0.60, description="Cosine similarity threshold (0.0 to 1.0)"),
    db: Session = Depends(get_db)
):
    """
    Detect all faces in the uploaded image, extract their embeddings,
    and search the database using cosine similarity via pgvector.
    """
    start_time = time.time()
    
    # Validate database connection
    try:
        db.execute(text("SELECT 1"))
    except Exception as db_err:
        logger.error(f"Database unavailable: {str(db_err)}")
        raise HTTPException(status_code=503, detail="Database unavailable. Please check backend connection.")

    # Retrieve InsightFace app
    face_app = getattr(request.app.state, "face_analysis", None)
    if face_app is None:
        raise HTTPException(status_code=500, detail="Face recognition model not loaded on server.")

    # Read image
    validate_upload(image)
    img = await read_image_from_upload(image)
    
    # Detect all faces
    inference_start = time.time()
    faces = detect_faces(face_app, img)
    inference_time = (time.time() - inference_start) * 1000
    
    logger.info(f"Search request: detected {len(faces)} face(s) in {inference_time:.1f}ms inference time")

    results = []
    
    for idx, face in enumerate(faces):
        bbox = [int(coord) for coord in face.bbox]
        
        try:
            embedding = extract_face_embedding(face)
        except Exception as emb_err:
            logger.warning(f"Failed to extract embedding for face {idx}: {str(emb_err)}")
            results.append({
                "face_index": idx,
                "bounding_box": bbox,
                "matched": False
            })
            continue
            
        # Search database
        search_start = time.time()
        matches = search_face_in_db(db, embedding, threshold=threshold)
        search_time = (time.time() - search_start) * 1000
        logger.info(f"Face {idx} database search completed in {search_time:.1f}ms")
        
        if matches:
            top_match = matches[0]
            matched_person = top_match["person"]
            similarity = top_match["similarity"]
            confidence_pct = round(similarity * 100, 2)
            
            # Gather reference images
            other_embeddings = db.query(FaceEmbedding).filter(
                FaceEmbedding.person_id == matched_person.id
            ).all()
            registered_images = [f"/static/{emb.image_path}" for emb in other_embeddings]
            
            person_response = MatchedPerson(
                id=matched_person.id,
                full_name=matched_person.full_name,
                case_number=matched_person.case_number,
                gender=matched_person.gender,
                age=matched_person.age,
                notes=matched_person.notes,
                created_at=matched_person.created_at,
                registered_images=registered_images
            )
            
            results.append({
                "face_index": idx,
                "bounding_box": bbox,
                "matched": True,
                "confidence": confidence_pct,
                "person": person_response
            })
            logger.info(f"Face {idx} matched: {matched_person.full_name} with confidence={confidence_pct}%")
        else:
            results.append({
                "face_index": idx,
                "bounding_box": bbox,
                "matched": False
            })
            logger.info(f"Face {idx} unmatched (below threshold)")

    processing_time = (time.time() - start_time) * 1000
    logger.info(f"Total search request processed in {processing_time:.1f}ms")
    return results


@router.get("/list", response_model=dict)
def list_registered_persons(db: Session = Depends(get_db)):
    """List all registered persons in the database with their metadata and references."""
    try:
        persons = db.query(Person).order_by(Person.created_at.desc()).all()
        result_list = []
        for person in persons:
            embs = db.query(FaceEmbedding).filter(FaceEmbedding.person_id == person.id).all()
            registered_images = [f"/static/{emb.image_path}" for emb in embs]
            result_list.append({
                "id": str(person.id),
                "full_name": person.full_name,
                "gender": person.gender,
                "age": person.age,
                "case_number": person.case_number,
                "notes": person.notes,
                "created_at": person.created_at.isoformat() if person.created_at else None,
                "registered_images": registered_images
            })
        return {
            "success": True,
            "persons": result_list
        }
    except Exception as e:
        logger.error(f"Error listing database faces: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Database list failed: {str(e)}")

