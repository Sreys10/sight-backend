import os
import sys
import cv2
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from routes.face_routes import router as face_router
from routes.forensic_routes import router as forensic_router
from database import init_db

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("face_recognition")

def download_opencv_models(root_dir: str):
    """
    Download OpenCV YuNet (face detection) and SFace (face recognition) ONNX models
    in chunks to keep memory usage low.
    """
    import urllib.request
    
    os.makedirs(root_dir, exist_ok=True)
    
    models = {
        # Official OpenCV Hugging Face repositories (bypasses Git LFS issue on opencv_zoo GitHub)
        "face_detection_yunet_2023mar.onnx": "https://huggingface.co/opencv/face_detection_yunet/resolve/main/face_detection_yunet_2023mar.onnx",
        "face_recognition_sface_2021dec.onnx": "https://huggingface.co/opencv/face_recognition_sface/resolve/main/face_recognition_sface_2021dec.onnx"
    }
    
    for filename, url in models.items():
        filepath = os.path.join(root_dir, filename)
        if not os.path.exists(filepath):
            logger.info(f"Downloading OpenCV face model '{filename}' from {url}...")
            tmp_filepath = filepath + ".tmp"
            try:
                with urllib.request.urlopen(url) as response, open(tmp_filepath, 'wb') as out_file:
                    chunk_size = 1024 * 1024  # 1MB chunks
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        out_file.write(chunk)
                os.rename(tmp_filepath, filepath)
                logger.info(f"✓ Model '{filename}' downloaded successfully.")
            except Exception as e:
                logger.error(f"Failed to download '{filename}': {str(e)}")
                if os.path.exists(tmp_filepath):
                    try:
                        os.remove(tmp_filepath)
                    except:
                        pass
        else:
            logger.info(f"✓ Model '{filename}' is already cached.")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup actions
    logger.info("Initializing Face Recognition & Forensic Backend...")
    
    # 1. Initialize PostgreSQL Database & pgvector Tables
    init_db()
    
    # 2. Get model configuration from environment
    model_root = os.getenv("OPENCV_MODEL_ROOT", "./.opencv_models")
    
    # 3. Initialize OpenCV YuNet & SFace models
    try:
        # Pre-download models if missing
        download_opencv_models(model_root)
        
        det_model_path = os.path.join(model_root, "face_detection_yunet_2023mar.onnx")
        rec_model_path = os.path.join(model_root, "face_recognition_sface_2021dec.onnx")
        
        if os.path.exists(det_model_path) and os.path.exists(rec_model_path):
            # Create FaceDetectorYN singleton (with standard default parameters)
            detector = cv2.FaceDetectorYN.create(
                model=det_model_path,
                config="",
                input_size=(320, 320),
                score_threshold=0.5,
                nms_threshold=0.3,
                top_k=5000
            )
            
            # Create FaceRecognizerSF singleton
            recognizer = cv2.FaceRecognizerSF.create(
                model=rec_model_path,
                config=""
            )
            
            app.state.face_detector = detector
            app.state.face_recognizer = recognizer
            logger.info("✓ OpenCV YuNet & SFace models successfully loaded.")
        else:
            logger.critical("✗ Required OpenCV face models are missing from cache.")
            app.state.face_detector = None
            app.state.face_recognizer = None
            
    except Exception as e:
        logger.critical(f"✗ Failed to load OpenCV face models: {str(e)}")
        app.state.face_detector = None
        app.state.face_recognizer = None
        
    yield
    
    # Shutdown actions
    logger.info("Shutting down Face Recognition & Forensic Backend...")

# Create FastAPI app
app = FastAPI(
    title="Evidence.ai Core API Service",
    description="FastAPI service mapping image tampering detection, weapon detection, metadata forensics, and pgvector face recognition.",
    version="1.0.0",
    lifespan=lifespan
)

# Set up CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the storage directory statically so uploaded face images are viewable
os.makedirs("storage", exist_ok=True)
app.mount("/static", StaticFiles(directory="storage"), name="static")

# Include routes
app.include_router(face_router)
app.include_router(forensic_router)

@app.get("/health")
def health_check():
    """Health check endpoint."""
    face_detector = getattr(app.state, "face_detector", None)
    face_recognizer = getattr(app.state, "face_recognizer", None)
    model_loaded = face_detector is not None and face_recognizer is not None
        
    return {
        "status": "healthy" if model_loaded else "degraded",
        "service": "face_recognition",
        "model_loaded": model_loaded,
        "backend": "OpenCV YuNet + SFace"
    }

if __name__ == "__main__":
    import uvicorn
    # Allow port to be configurable via PORT environment variable, defaults to 8000
    # In production, Render/Railway injects PORT
    port = int(os.getenv("PORT", 5001))
    logger.info(f"Starting server on port {port}...")
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
